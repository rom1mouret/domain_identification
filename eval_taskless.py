#!/usr/bin/env python3

from typing import Tuple
import sys
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm, trange

from nn_utils import freeze, masked_mean
from recipe_dataset import Dataset
from eval_utils import scoring, measure_time
from architecture import Processor, AbstractToGoal, AbstractToAbstract

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print("running on device:", device)

# architecture
random_domain = 1907
vocab_size = 4096
abstraction_dim = int(sys.argv[1])
predictor = AbstractToGoal(vocab_size=vocab_size, abstraction_dim=abstraction_dim).to(device)
abstract_to_abstract = AbstractToAbstract(abstraction_dim).to(device)

# Data
domains = list(range(7))
experiment_name = "%i domains [dim=%i] P" % (len(domains), abstraction_dim)
train_data0, train_data1, train_data2, test_data = \
    Dataset(sys.argv[2:], min_len=2, rm_duplicates=True).split(0.4, 0.45, 0.1, 0.05)

# train_data0: for predictor and abstract_to_abstract
# train_data1: for processors
# train_data2: fine-tuning of abstract_to_abstract and evaluating processors


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
loss_func = nn.CrossEntropyLoss(reduction='none')
system = nn.ModuleList([processor, predictor, abstract_to_abstract])
optimizer = torch.optim.Adam(system.parameters(), lr=0.001)
with measure_time("abstraction building"):
    running_loss = 1
    progress_bar = trange(2)
    for epoch in progress_bar:
        train_data0.reset()
        while not train_data0.empty():
            sentences = train_data0.get(64)
            batch, y_true, _ = io(sentences, domain=random_domain)
            abstraction = processor(batch)
            task_pred = predictor(abstraction)
            abstract_pred = abstract_to_abstract(abstraction)
            dist = (abstract_pred[:, :, :-1] - abstraction[:, :, 1:]).pow(2).mean(dim=1)
            amplitude = torch.tanh(abstraction).pow(2).mean(dim=1)
            entropy = loss_func(task_pred, y_true)
            loss = masked_mean(dist, sentences) + 1 + masked_mean(entropy - amplitude, sentences)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss = 0.99 * running_loss + (1 - 0.99) * loss.item()
            progress_bar.set_description("loss: %.5f" % running_loss)

for net in (processor, predictor):
    freeze(net)

# fine-tune abstract_to_abstract
optimizer = torch.optim.Adam(abstract_to_abstract.parameters(), lr=0.001)
with measure_time("fine tuning"):
    running_loss = 1
    train_data2.reset()
    with trange(1) as progress_bar:
        while not train_data2.empty():
            sentences = train_data2.get(64)
            batch, y_true, _ = io(sentences, domain=random_domain)
            with torch.no_grad():
                abstraction = processor(batch)
            abstract_pred = abstract_to_abstract(abstraction)
            dist = (abstract_pred[:, :, :-1] - abstraction[:, :, 1:]).pow(2).mean(dim=1)
            loss = masked_mean(dist, sentences)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss = 0.99 * running_loss + (1 - 0.99) * loss.item()
            progress_bar.set_description("loss: %.5f" % running_loss)

freeze(abstract_to_abstract)

# main training
processors = []
processor_to_domain = []
progress_bar = tqdm(domains, total=len(domains))
for domain in progress_bar:
    with measure_time("neural network training"):
        train_data1.reset()
        n = 0
        while not train_data1.empty():
            n += 1
            if n % 1000 == 1:
                print("last loss", loss.item())
                processor = Processor(
                                vocab_size=vocab_size,
                                abstraction_dim=abstraction_dim).to(device)
                optimizer = torch.optim.Adam(processor.parameters(), lr=0.01)
                processors.append(processor)
                processor_to_domain.append(domain)

            sentences = train_data1.get(32)
            if len(sentences) < 10:
                continue
            batch, y_true, _ = io(sentences, domain=domain)
            abstraction = processor(batch)
            task_pred = predictor(abstraction)
            abstract_pred = abstract_to_abstract(abstraction)
            dist = (abstract_pred[:, :, :-1] - abstraction[:, :, 1:]).pow(2).mean(dim=1)
            entropy = loss_func(task_pred, y_true)
            loss = masked_mean(entropy, sentences) + masked_mean(dist, sentences)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

for processor in processors:
    processor.eval()


# remove poorly-performing processors
kept = [
    i for i, domain in enumerate(processor_to_domain[:-1])
    if domain == processor_to_domain[i+1]
]
processors = [processors[i] for i in kept]
processor_to_domain = [processor_to_domain[i] for i in kept]
scores = [0] * len(kept)

print("number of processors:", len(processors))


with scoring(experiment_name, "domain_accuracy_taskless.txt") as accuracy:
    for correct_domain in trange(len(domains)):
        test_data.reset(shuffle=False)
        while not test_data.empty():
            # small batch to fit into 6GB of video RAM with 50+ processors
            sentences = test_data.get(16, min_len=6)
            batch, y_true, _ = io(sentences, correct_domain)
            # run all processors
            processed = [
                processor(batch) for processor in processors
            ]
            pred = torch.cat([
                abstract_to_abstract(abstract).unsqueeze(3)
                for abstract in processed
            ], dim=3)
            abstract = torch.cat([x.unsqueeze(3) for x in processed], dim=3)
            # align predictions
            abstract = abstract[:, :, :-1, :]  # the last one has no counterpart
            pred = pred[:, :, 1:, :]  # the first one cannot be predicted by abstract_to_abstract
            # minimum distance
            dist = (pred - abstract).pow(2).sum(dim=1)
            for j, sentence in enumerate(sentences):
                seq_dist = dist[j, :len(sentence), :].log().sum(dim=0) #before: prod
                _, arg_min_dist = seq_dist.min(dim=0)  # best processor
                if processor_to_domain[arg_min_dist.item()] == correct_domain:
                    accuracy.register(100)
                else:
                    accuracy.register(0)


# Sanity check. This is optional:
# with scoring(experiment_name) as accuracy:
#     with scoring(experiment_name) as proba:
#         for processor, domain in tqdm(zip(processors, processor_to_domain), total=len(processors)):
#             test_data.reset(shuffle=False)
#             while not test_data.empty():
#                 sentences = test_data.get(1024)
#                 batch, y_true, np_y_true = io(sentences, domain)
#                 y_pred = predictor(processor(batch))
#                 best = y_pred.max(dim=1)[1].cpu().numpy()
#                 # measure accuracy
#                 match = best == np_y_true
#                 for j, sentence in enumerate(sentences):
#                     for k in range(len(sentence)):
#                         if match[j, k]:
#                             accuracy.register(100)
#                         else:
#                             accuracy.register(0)
#                 # measure probability of target
#                 probas = torch.softmax(y_pred, dim=1).cpu().numpy()
#                 for j, sentence in enumerate(sentences):
#                     for k in range(len(sentence)):
#                         p = probas[j, np_y_true[j, k], k]
#                         proba.register(100*p)
