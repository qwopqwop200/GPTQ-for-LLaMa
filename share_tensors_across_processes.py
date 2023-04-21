import gc, os, pickle

import accelerate, torch, tqdm

accelerate_load_checkpoint_and_dispatch = accelerate.load_checkpoint_and_dispatch

def load_checkpoint_shared_and_dispatch(model, checkpoint, device_map = 'auto', max_memory = None, no_split_module_classes = None): #**kwparams):
    try:
        with open(f'{checkpoint}.shared', 'rb') as shared:
            shared = pickle.load(shared)
            with tqdm.tqdm(
                shared['state_dict'].items(),
                desc = 'Importing shared tensors ...',
                unit = 'w',
                leave = False,
            ) as shared_tensors:
                os.kill(shared['pid'], 0) # raises if pid does not exist
                device_map = shared['device_map']
                #offload_dir = shared['offload_folder']
                #offload_buffers = shared['offload_buyffers']
                #preload_module_classes = shared['preload_module_classes']
                state_dict = {
                    name: rebuild_tensor(*tensor_params)
                    for name, (rebuild_tensor, tensor_params)
                    in shared_tensors
                }
        model.load_state_dict(state_dict, strict=False)
        for param_name, param in state_dict.items():
            module_name = param_name

            while len(module_name) > 0 and module_name not in device_map:
                module_name = '.'.join(module_name.split('.')[:-1])
            param_device = device_map[module_name]

            #if param_device == 'disk':
            #
            accelerate.utils.modeling.set_module_tensor_to_device(model, param_name, param_device, value=param)#, **kwparams)
    except (FileNotFoundError, EOFError, KeyError, ProcessLookupError, RuntimeError):
        if device_map != 'sequential':
            max_memory = accelerate.utils.get_balanced_memory(
                model,
                max_memory=max_memory,
                no_split_module_classes=no_split_module_classes,
                low_zero=(device_map == 'balanced_low_0'),
                #**kwparams,
            )
        if isinstance(device_map, str):
            device_map = accelerate.infer_auto_device_map(
                model, max_memory=max_memory, no_split_module_classes=no_split_module_classes#, **kwparams
            )
        #if not kwparams.get('offload_state_dict') and device_map is not None and 'disk' in device_map.values():
        #    offload_state_dict = True
        accelerate.load_checkpoint_in_model(model, checkpoint, device_map=device_map)#, **kwparams)
        state_dict = model.state_dict()
        with open(f'{checkpoint}.shared', 'wb') as shared, tqdm.tqdm(
                    state_dict.items(),
                    desc = 'Exporting shared tensors ...',
                    unit = 'w',
                    ) as state_dict_items:
            pickle.dump({
                'pid': os.getpid(),
                'device_map': device_map,
                'state_dict': {
                    name: torch.multiprocessing.reductions.reduce_tensor(tensor.share_memory_())
                    for name, tensor in state_dict_items
                }
            }, shared)

    del state_dict
    gc.collect()

    if device_map is not None:
        model = accelerate.big_modeling.dispatch_model(model, device_map=device_map)#, **kwparams)
    return model

accelerate.load_checkpoint_and_dispatch = load_checkpoint_shared_and_dispatch
