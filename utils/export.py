import numpy  as np
import toml

def export_quant_table(quantizers: dict, quant_dir:str, format:str = 'toml'):

    table = {}

    def save_tensor(name: str, tensor):
        np.save(name, tensor.numpy())
        return '{}.npy'.format(name)

    for key, value in quantizers.items():
        quantizer = value[0]

        dump = dict()

        sym = quantizer.sym
        if not sym:
            dump['zero'] = save_tensor(value[2])
        
        dump['scale'] = save_tensor(value[1])
        dump['wbits'] = value[4]
        dump['groupsize'] = value[5]
        if value[5] > 0:
            dump['group_ids'] = save_tensor(group_ids)

        dump['sym'] = sym
        dump['perchannel'] = quantizer.perchannel
        
        decoder_id = 'decoder.{}'.format(key.split('.')[2])
        if decoder_id not in table:
            table[decoder_id] = {}
        table[decoder_id][key] = dump

    # for k, value in table:
    if not os.path.exists(quant_dir):
        os.mkdir(quant_dir)
    
    for key, value in table.items():
        with open(os.path.join(quant_dir, '{}.toml'.format(key)), 'w') as f:
            toml.dump(value, f)
