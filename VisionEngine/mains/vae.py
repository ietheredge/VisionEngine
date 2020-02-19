def make_vlae(latent_size, use_perceptual_loss=False):
    with tf.name_scope('encoder'):
        encoder = make_encoder(latent_size)
    with tf.name_scope('decoder'):
        decoder = make_decoder(latent_size, latent_size, latent_size, latent_size)
    inputs = Input((256,256,3))
    h_1, h_2, h_3, h_4 = encoder(inputs)
    z_1 = NormalVariational(latent_size, add_kl=False, coef_kl=1., add_mmd=True, lambda_mmd=5000., name='z_1_latent')(h_1)
    z_2 = NormalVariational(latent_size, add_kl=False, coef_kl=1., add_mmd=True, lambda_mmd=5000., name='z_2_latent')(h_2)
    z_3 = NormalVariational(latent_size, add_kl=False, coef_kl=1., add_mmd=True, lambda_mmd=5000., name='z_3_latent')(h_3)
    z_4 = NormalVariational(latent_size, add_kl=False, coef_kl=1., add_mmd=True, lambda_mmd=5000., name='z_4_latent')(h_4)
    decoded = decoder([z_1, z_2, z_3, z_4])
    vlae = Model(inputs, decoded, name='vlae')
    return vlae