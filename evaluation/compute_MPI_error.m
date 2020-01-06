clear all;
totalMSEA = 0;
totalLMSEA = 0;
totalDSSIMA = 0;
totalMSES = 0;
totalLMSES = 0;
totalDSSIMS = 0;
count = 0;

% reflect_path = 'D:/fangyang/intrinsic_by_fangyang/logs_shapenet/RIN_CosBF_VGG0.1/refl_target/'
% shading_path = 'D:/fangyang/intrinsic_by_fangyang/logs_shapenet/RIN_CosBF_VGG0.1/shad_target/'
% result_reflect_path = 'D:/fangyang/intrinsic_by_fangyang/logs_shapenet/RIN_CosBF_VGG0.1/refl_output/';
% result_shading_path = 'D:/fangyang/intrinsic_by_fangyang/logs_shapenet/RIN_CosBF_VGG0.1/shad_output/';
% mask_path = 'D:/fangyang/intrinsic_by_fangyang/logs_shapenet/RIN_CosBF_VGG0.1_updateLR/mask/'
Dir = 'F:/BOLD/test_result_addShapeCosNoShadingCosBFVGG0.1/'
% reflect_path = [ Dir 'refl_target_fullsize/'];
% shading_path = [ Dir 'shad_target_fullsize/'];
% result_reflect_path = [ Dir 'refl_output_fullsize/'];
% result_shading_path = [ Dir 'shad_output_fullsize/'];
reflect_path = [ Dir 'refl_target_fullsize/'];
shading_path = [ Dir 'shad_target_fullsize/'];
result_reflect_path = [ Dir 'refl_output_fullsize/'];
result_shading_path = [ Dir 'shad_output_fullsize/'];

fileFolder = fullfile(reflect_path);
dirOutput = dir(fullfile(fileFolder,'*.png'));
fileNames = {dirOutput.name}';

% s = dir([inputDir '*-input.png']);
for i = 1:length(fileNames)
    
    im_reflect_path = strcat(reflect_path,fileNames(i));
    im_reflect_path = im_reflect_path{1};
    im_shading_path = strcat(shading_path,fileNames(i));
    im_shading_path = im_shading_path{1};
%     im_mask_path = strcat(mask_path,fileNames(i));
%     im_mask_path =im_mask_path{1};
    im_reflect_result_path = strcat(result_reflect_path,fileNames(i));
    im_reflect_result_path = im_reflect_result_path{1};
    im_shading_result_path = strcat(result_shading_path,fileNames(i));
    im_shading_result_path = im_shading_result_path{1};
    
    albedoName = im_reflect_result_path;
    shadingName = im_shading_result_path;
    labelAlbedoName = im_reflect_path;
    labelShadingName = im_shading_path;
%     maskName = im_mask_path;
    
%     disp(sprintf('name: %s', albedoName));

    albedo = im2double(imread(albedoName));
    labelAlbedo = im2double(imread(labelAlbedoName));
    shading = im2double(imread(shadingName));
    labelShading = im2double(imread(labelShadingName));
%     mask = im2double(imread(maskName));
    
%     albedo = albedo .* mask;
%     labelAlbedo = labelAlbedo .* mask;
%     shading = shading .* mask;
%     labelShading = labelShading .* mask;
    
    [height, width, channel] = size(albedo);

    totalMSEA = totalMSEA + evaluate_one_k(albedo,labelAlbedo);
%     disp(sprintf('albedo: mse: %f', totalMSEA));
    totalLMSEA = totalLMSEA + levaluate_one_k(albedo,labelAlbedo);
    totalDSSIMA = totalDSSIMA + (1-evaluate_ssim_one_k_fast(albedo,labelAlbedo))/2;

    totalMSES = totalMSES + evaluate_one_k(shading,labelShading);
    totalLMSES = totalLMSES + levaluate_one_k(shading,labelShading);
    totalDSSIMS = totalDSSIMS + (1-evaluate_ssim_one_k_fast(shading,labelShading))/2;
    
    count = count + 1;
    disp(count);
end
totalMSEA = totalMSEA/count;
totalLMSEA = totalLMSEA/count;
totalDSSIMA = totalDSSIMA/count;
totalMSES = totalMSES/count;
totalLMSES = totalLMSES/count;
totalDSSIMS = totalDSSIMS/count;
disp(sprintf('albedo: mse: %f, lmse: %f, dssim: %f',totalMSEA,totalLMSEA,totalDSSIMA));
disp(sprintf('shading: mse: %f, lmse: %f, dssim: %f',totalMSES,totalLMSES,totalDSSIMS));