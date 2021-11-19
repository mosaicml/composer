import pickle


def disable_dropout(arch, net):
    # manually turn off dropout to pass backward check
    # TODO(laura): make model-agnostic
    if arch in ['alexnet']:
        net.classifier[0].p = 0
        net.classifier[3].p = 0
    elif arch in ['vgg11', 'vgg13', 'vgg16', 'vgg19']:
        net.classifier[2].p = 0
        net.classifier[5].p = 0
    elif arch in ['inception_v3']:
        net.drop_rate = 0
    elif arch in ['densenet121', 'densenet161', 'densenet169', 'densenet201']:
        for i, num_layers in enumerate(net.block_config):
            block = net.features.__getattr__('denseblock%d' % (i + 1))
            for i in range(num_layers):
                layer = block.__getattr__('denselayer%d' % (i + 1))
                layer.drop_rate = 0
    elif arch in ['darts_cifar10', 'nasnet_cifar10', 'amoebanet_cifar10']:
        net.drop_path_prob = 0
    else:
        pass


def add_vertex_cost_to_edge(G):
    for edge_key in G.edges:
        _, target_key, _ = edge_key
        target_cost = G.nodes[target_key]['cost']
        edge_cost = G.edges[edge_key]['cost']
        G.edges[edge_key]['weight'] = target_cost + edge_cost

    return G


def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def save_pickle(path, data):
    with open(path, 'wb') as f:
        pickle.dump(data, f)


def forward_backward(module, device, input_size=(1, 3, 224, 224), repeat=100, min_repeat=5):

    input2 = torch.rand(*input_size, device=device)
    input2.requires_grad = True
    output2 = module(input2)
    loss = torch.sum(output2)
    loss.backward()
    del input2, output2, loss

    torch.cuda.reset_max_memory_allocated(device)
    regular_start_memory = torch.cuda.max_memory_allocated(device)
    regular_times = []

    for i in tqdm(range(repeat)):
        input2 = torch.rand(*input_size, device=device)
        input2.requires_grad = True
        start = time.time()
        output2 = module(input2)
        loss = torch.sum(output2)
        loss.backward()
        end = time.time()
        regular_times.append(end - start)
        del input2, output2, loss

    regular_peak_memory = torch.cuda.max_memory_allocated(device)

    regular_end_memory = torch.cuda.memory_allocated(device)
    regular_avg_time = np.mean(np.array(regular_times)[min_repeat:])

    torch.cuda.empty_cache()

    return regular_start_memory, regular_end_memory, regular_peak_memory, regular_avg_time


def forward_backward_benchmark(net, run_segment, device, input_size=(1, 3, 224, 224), repeat=100, min_repeat=5):
    assert repeat > min_repeat
    net.train()

    regular_start_memory, regular_end_memory, regular_peak_memory, regular_avg_time = forward_backward(
        net, device, input_size, repeat, min_repeat)
    checkpoint_start_memory, checkpoint_end_memory, checkpoint_peak_memory, checkpoint_avg_time = forward_backward(
        run_segment, device, input_size, repeat, min_repeat)

    regular_pytorch_overhead = max(regular_start_memory, regular_end_memory)
    checkpoint_pytorch_overhead = max(
        checkpoint_start_memory, checkpoint_end_memory)

    regular_intermediate_tensors = regular_peak_memory - regular_pytorch_overhead
    checkpoint_intermediate_tensors = checkpoint_peak_memory - checkpoint_pytorch_overhead

    print('Average Iteration Time: Checkpointing {:.4f} s, Regular {:.4f} s, overhead {:.2f}%'.format(
        checkpoint_avg_time, regular_avg_time, (checkpoint_avg_time - regular_avg_time) * 100 / regular_avg_time))
    print('Average Peak Memory: Checkpointing {:.4f} MB, Regular {:.4f} MB, Memory Cut off {:.2f}%'.format(
        checkpoint_peak_memory / (1024**2), regular_peak_memory / (1024**2), (regular_peak_memory - checkpoint_peak_memory) * 100 / regular_peak_memory))
    print('Average Intermediate Tensors: Checkpointing {:.4f} MB, Regular {:.4f} MB, Memory Cut off {:.2f}%'.format(
        checkpoint_intermediate_tensors / (1024 ** 2), regular_intermediate_tensors / (1024 ** 2), (regular_intermediate_tensors - checkpoint_intermediate_tensors) * 100 / regular_intermediate_tensors))
