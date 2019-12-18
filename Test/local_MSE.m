function local_error = local_MSE(img_GT,img_es,mask,window_size)

    if ~exist('window_size','var')
        window_size = 20;
    end
    if ~exist('mask','var')
        mask = all(img_GT>0, 3);
        mask = double(mask);
    end
    img_GT=im2double(img_GT);
    img_es=im2double(img_es);
    mask=double(mask);
    

    [img_M img_N img_C] = size(img_GT);
    shift = ceil(window_size/2);
    total = 0;
    ssq = 0;    
    for k = 1:img_C
        for i = 1 : shift : img_M-window_size+1
            for j = 1 : shift : img_N-window_size+1
                correct_curr = img_GT(i:i+window_size-1,j:j+window_size-1,k);
                estimate_curr = img_es(i:i+window_size-1,j:j+window_size-1,k);
                mask_curr = mask(i:i+window_size-1,j:j+window_size-1);
                ssq = ssq + ssq_error(correct_curr, estimate_curr, mask_curr);
                total = total + sum(sum(mask_curr .* correct_curr .^ 2));
            end
        end
    end
    if(isnan(ssq/total))
        local_error = 0;
    else
        local_error = ssq/total;
    end
    
end


function ssq = ssq_error(img_GT,img_es,mask)
    if sum(sum(img_es .^ 2 .* mask))>1e-5
        alpha = sum(sum(img_GT .* img_es .* mask))/ sum(sum(img_es .^ 2 .* mask));
    else
        alpha = 0;
    end
    ssq = sum(sum(mask .* (img_GT - alpha .* img_es) .^ 2));
end

