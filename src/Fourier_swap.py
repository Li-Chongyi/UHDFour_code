 #haze_down=F.interpolate(haze,scale_factor=0.125,mode='bilinear')
            # gt_down=F.interpolate(gt,scale_factor=0.25,mode='bilinear')

            # x_freq = torch.fft.rfft2(haze, norm='backward')
            
            # y_freq = torch.fft.rfft2(gt, norm='backward')
            
            # mag_x = torch.abs(x_freq)
            # pha_x = torch.angle(x_freq)

            # mag_y = torch.abs(y_freq)
            # pha_y = torch.angle(y_freq)

            # real = mag_x * torch.cos(pha_y)
            # imag = mag_x * torch.sin(pha_y)
            # x_out = torch.complex(real, imag)
            # x_freq_spatial = torch.fft.irfft2(x_out, norm='backward')
            
            # a = sns.heatmap(torch.squeeze(mag_x, 0)[0,:,:].cpu(), xticklabels=False, yticklabels=False, cbar=True, square=True, cmap='viridis')
            # plt.savefig('./heatmap.png',  bbox_inches='tight', pad_inches=0)
            # save_image(x_freq_spatial, image_name, category)
            # import pdb
            # pdb.set_trace()  