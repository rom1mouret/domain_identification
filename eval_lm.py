#!/usr/bin/env python3

from typing import Tuple
import sys
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm, trange

from nn_utils import freeze
from recipe_dataset import Dataset
from abstract_machine import AbstractMachine
from eval_utils import scoring, measure_time
from architecture import Processor, AbstractToGoal, AbstractToAbstract

device = "cuda:0"

# architecture
vocab_size = 4096
abstraction_dim = int(sys.argv[1])
predictor = AbstractToGoal(vocab_size=vocab_size, abstraction_dim=abstraction_dim).to(device)
abstract_to_abstract = AbstractToAbstract(abstraction_dim).to(device)

# Data
domains = list(range(7))
experiment_name = "%i domains [dim=%i] P" % (len(domains), abstraction_dim)
train_data1, train_data2, test_data = Dataset(sys.argv[2:]).split(0.75, 0.2, 0.05)

def io(sentences: list, domain: int) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    max_len = max(map(len, sentences))
    np_packed = np.zeros((len(sentences), max_len), dtype=np.int64)
    for i, sentence in enumerate(sentences):
        for j, w in enumerate(sentence):
            np_packed[i, j] = w + 1  # 0 is reserved for 'empty'

    packed = torch.from_numpy(np_packed).to(device)
    batch = (packed + domain) % vocab_size
    y_true = packed % vocab_size
    batch[batch == domain] = 0  # empty
    np_packed %= vocab_size  # numpy version of y_true

    return batch.to(device), y_true.to(device), np_packed


# find good abstractions on a random domain
processor = Processor(vocab_size=vocab_size, abstraction_dim=abstraction_dim).to(device)
loss_func = nn.CrossEntropyLoss()
system = nn.ModuleList([processor, predictor, abstract_to_abstract])
optimizer = torch.optim.Adam(system.parameters(), lr=0.001)
with measure_time("abstraction building"):
    running_loss = 1
    progress_bar = trange(2)
    for epoch in progress_bar:
        train_data1.reset()
        while not train_data1.empty():
            sentences = train_data1.get(64)
            batch, y_true, _ = io(sentences, domain=1907)
            abstraction = processor(batch)
            task_pred = predictor(abstraction)
            abstract_pred = abstract_to_abstract(abstraction)
            dist = (abstract_pred[:, :, :-1] - abstraction[:, :, 1:]).pow(2).mean()
            amplitude = torch.tanh(abstraction).pow(2).mean()
            loss = loss_func(task_pred, y_true) + dist + 1 - amplitude
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss = 0.99 * running_loss + (1 - 0.99) * loss.item()
            progress_bar.set_description("loss: %.5f" % running_loss)

for net in (processor, predictor):
    freeze(predictor)

# fine-tune abstract_to_abstract
optimizer = torch.optim.Adam(abstract_to_abstract.parameters(), lr=0.001)
with measure_time("fine tuning"):
    running_loss = 1
    train_data2.reset()
    with trange(1) as progress_bar:
        while not train_data2.empty():
            sentences = train_data2.get(64)
            batch, y_true, _ = io(sentences, domain=1907)
            with torch.no_grad():
                abstraction = processor(batch)
            abstract_pred = abstract_to_abstract(abstraction)
            loss = (abstract_pred[:, :, :-1] - abstraction[:, :, 1:]).pow(2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss = 0.99 * running_loss + (1 - 0.99) * loss.item()
            progress_bar.set_description("loss: %.5f" % running_loss)

freeze(abstract_to_abstract)

# training
processors = []
loss_func = nn.CrossEntropyLoss()
progress_bar = tqdm(domains, total=len(domains))
for domain in progress_bar:
    with measure_time("neural network training"):
        best_loss = np.inf
        for attempt in range(3):
            processor = Processor(
                            vocab_size=vocab_size,
                            abstraction_dim=abstraction_dim).to(device)
            optimizer = torch.optim.Adam(processor.parameters(), lr=0.005)
            running_loss = 1
            train_data1.reset()
            while not train_data1.empty():
                sentences = train_data1.get(64)
                batch, y_true, _ = io(sentences, domain)
                abstraction = processor(batch)
                task_pred = predictor(abstraction)
                abstract_pred = abstract_to_abstract(abstraction)
                dist = (abstract_pred[:, :, :-1] - abstraction[:, :, 1:]).pow(2).mean()
                loss = loss_func(task_pred, y_true) + dist
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss = 0.99 * running_loss + (1 - 0.99) * loss.item()
                progress_bar.set_description("loss: %.5f" % running_loss)
            if running_loss < best_loss:
                best_loss = running_loss
                best_processor = processor
            if running_loss < 0.95:
                break  # good enough
        processors.append(best_processor.eval())

# domain identification evaluation
with scoring(experiment_name, "domain_accuray.txt") as accuracy:
    for correct_domain in trange(len(processors)):
        processor = processors[correct_domain]
        test_data.reset(shuffle=False)
        while not test_data.empty():
            sentences = test_data.get(1024)
            batch, y_true, _ = io(sentences, correct_domain)
            # reference abstraction of shape
            # n_sentences x abstraction_dim x sentence_max_length
            correct_abstraction = processor(batch)
            # predicted abstraction
            pred = abstract_to_abstract(correct_abstraction)
            # align predictions with sentences
            pred = pred[:, :, :-1]
            batch = batch[:, 1:]
            # run all processors
            processed = torch.cat([
                processor(batch).unsqueeze(3) for processor in processors
            ], dim=3)
            # dist
            dist = (pred.unsqueeze(3) - processed).pow(2).sum(dim=1)
            min_dist = dist.min(dim=2)[1].cpu().numpy()
            # measure domain selection accuracy
            for j, sentence in enumerate(sentences):
                for k in range(len(sentence)-1):
                    if min_dist[j, k] == correct_domain:
                        accuracy.register(100)
                    else:
                        accuracy.register(0)

# LM evaluation
with scoring(experiment_name, "LM_accuracy.txt") as accuracy:
    with scoring(experiment_name, "LM_target_proba.txt") as proba:
        for domain, processor in enumerate(tqdm(processors, total=len(processors))):
            test_data.reset(shuffle=False)
            while not test_data.empty():
                sentences = test_data.get(1024)
                batch, y_true, np_y_true = io(sentences, domain)
                y_pred = predictor(processor(batch))
                best = y_pred.max(dim=1)[1].cpu().numpy()
                # measure accuracy
                match = best == np_y_true
                for j, sentence in enumerate(sentences):
                    for k in range(len(sentence)):
                        if match[j, k]:
                            accuracy.register(100)
                        else:
                            accuracy.register(0)
                # measure probability of target
                probas = torch.softmax(y_pred, dim=1).cpu().numpy()
                for j, sentence in enumerate(sentences):
                    for k in range(len(sentence)):
                        p = probas[j, np_y_true[j, k], k]
                        proba.register(100*p)
