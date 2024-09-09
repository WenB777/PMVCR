def get_default_config(data_name):
    if data_name in ['BDGP']:
        return dict(
            Autoencoder=dict(
                arch0=[1750, 1024, 1024, 1024, 128],
                arch1=[79, 1024, 1024, 1024, 128],
                activations0='relu', activations1='relu', batchnorm=True
            ),
            training=dict(
                view=2, lambda11=10, lambda12=100, lambda13=1, lambda14=0.01, 
                lambda21=10, lambda22=0.01, lambda23=100,
                batch_size=256, n_class=5, temper=1
            )
        )
    elif data_name in ['Reuters_dim10']:
        return dict(
            Autoencoder=dict(
                arch0=[10, 1024, 1024, 1024, 128],
                arch1=[10, 1024, 1024, 1024, 128],
                activations0='relu', activations1='relu', batchnorm=True
            ),
            training=dict(
                view=2, lambda11=10, lambda12=100, lambda13=10,lambda14=0.1, 
                lambda21=10, lambda22=0.1, lambda23=100,
                batch_size=256, n_class=6, temper=0.5, temper1=1
            )
        )
    elif data_name in ['Scene15']:
        return dict(
            Autoencoder=dict(
                arch0=[20, 1024, 1024, 1024, 128],
                arch1=[59, 1024, 1024, 1024, 128],
                activations0='relu', activations1='relu', batchnorm=True
            ),
            training=dict(
                view=2, lambda11=10, lambda12=10, lambda13=0.01,lambda14=0.1, 
                lambda21=0.01, lambda22=10, lambda23=0.01,
                batch_size=256, n_class=15, temper=0.5, temper1=1, seed=7
            )
        )
    elif data_name in ['RGBD']:
        return dict(
            Autoencoder=dict(
                arch0=[2048, 1024, 1024, 1024, 32],
                arch1=[300, 1024, 1024, 1024, 32],
                activations0='relu', activations1='relu', batchnorm=True
            ),
            training=dict(
                view=2, lambda11=10, lambda12=100, lambda13=10,lambda14=0.01, 
                lambda21=100, lambda22=0.1, lambda23=100,
                batch_size=512, n_class=13, temper=0.5, seed=7
            )
        )
    else:
        raise Exception('Undefined data_name')
