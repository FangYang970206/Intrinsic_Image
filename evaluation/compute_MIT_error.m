clear all;
Dir = 'D:\fangyang\intrinsic_by_fangyang\MIT_logs\GAN_RIID_updateLR3_epoch160_CosbfVGG_refl-se-skip_shad-se-low_multi_new_shadSqueeze_DA_256_MITError\';
albedo_pred_dir = [Dir 'refl_output\'];
albedo_targ_dir = [Dir 'refl_target\'];
shading_pred_dir = [Dir 'shad_output\'];
shading_targ_dir = [Dir 'shad_target\'];
mask_dir = [Dir 'mask\'];
images = dir([albedo_pred_dir '*.png']);
mse_albedo = {};
mse_shading = {};
lmse = {};

for m =1:length(images)
    albedoname_predict = [albedo_pred_dir num2str(m - 1) '.png'];
    shadingname_predict = [shading_pred_dir num2str(m - 1) '.png'];
    albedoname_label = [albedo_targ_dir num2str(m - 1) '.png'];
    shadingname_label = [shading_targ_dir num2str(m - 1) '.png'];
    maskname_label = [mask_dir num2str(m - 1) '.png'];
    
    albedo_predict = im2double(imread(albedoname_predict));
    shading_predict = im2double(imread(shadingname_predict));
    albedo_label = im2double(imread(albedoname_label));
    shading_label = im2double(imread(shadingname_label));
    mask = (imread(maskname_label));
    mask = mask(:, :, 1);
    V = mask > 0;

    V3 = repmat(V,[1,1,size(shading_label,3)]);  
    
    errs_grosse = nan(1, size(albedo_label,3));
    for c = 1:size(albedo_label,3)
      errs_grosse(c) = 0.5 * MIT_mse(shading_predict(:,:,c), shading_label(:,:,c), V) + 0.5 * MIT_mse(albedo_predict(:,:,c), albedo_label(:,:,c), V);
    end
    lmse{m} = mean(errs_grosse);
    
    alpha_shading = sum(shading_label(V3) .* shading_predict(V3)) ./ max(eps, sum(shading_predict(V3) .* shading_predict(V3)));
    S = shading_predict * alpha_shading;
%     tmp1 = albedo_label(V3) .* albedo_predict(V3);
    alpha_reflectance = sum(albedo_label(V3) .* albedo_predict(V3)) ./ max(eps, sum(albedo_predict(V3) .* albedo_predict(V3)));
    A = albedo_predict * alpha_reflectance;
%     disp(sprintf('albedo: mse: %f', sum(sum(sum(A)))));

    mse_shading{m} =  mean((S(V3) - shading_label(V3)).^2);
    mse_albedo{m} =  mean((A(V3) - albedo_label(V3)).^2);
end

ave_lmse = 0;
ave_mse_albedo = 0;
ave_mse_shading = 0;
for m =1:length(images)
    ave_lmse = ave_lmse + log(lmse{m});
    ave_mse_albedo = ave_mse_albedo + log(mse_albedo{m});
    ave_mse_shading = ave_mse_shading + log(mse_shading{m});
end
ave_lmse = exp(ave_lmse/length(images));
ave_mse_albedo = exp(ave_mse_albedo/length(images));
ave_mse_shading = exp(ave_mse_shading/length(images));
disp(sprintf('ave_lmse: %f\nave_mse_albedo: %f\nave_mse_shading: %f\n',ave_lmse,ave_mse_albedo,ave_mse_shading));