def build_model(opt):
    if opt['network_type'] == "NLformer":
        from model.nlformer_arch import NLformer
        model_restoration = NLformer(channel=opt['n_channel'], win=opt['win'], ln_bias=opt['ln_bias'], dim=opt['dim'], ffn_factor=opt['ffn_factor'], head_num=opt['head_num'], num_blocks=opt['num_blocks'], num_refine_blocks=opt['num_refine_blocks'])

    else:
        raise NotImplementedError("Network [{:s}] is not found.".format(opt['network_type']))
    
    return model_restoration