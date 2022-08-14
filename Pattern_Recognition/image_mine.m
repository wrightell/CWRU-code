function glcm = image_mine(data,labels,num_levels,offset)
    
    %coarsen the images
    C = coarsen_mat(data,8);
    
    %get the size
    [s1,s2] = size(data);

    %intialize to 0
    glcm = zeros(5*num_levels^2,s2);
    
    %for each image, transform it into a glcm vector k^2 x 1
    for i = 1:length(data(1,:))
    
        img = reshape(C(:,i),128,128)';
    
        %try with 3 or 5 different lengths
        glcm_1 = reshape(GLCM(img,offset,0,num_levels,sqrt(s1)),1,num_levels^2);
        glcm_2 = reshape(GLCM(img,0,offset,num_levels,sqrt(s1)),1,num_levels^2);
        glcm_3 = reshape(GLCM(img,offset,offset,num_levels,sqrt(s1)),1,num_levels^2);
        glcm_4 = reshape(GLCM(img,2*offset,offset,num_levels,sqrt(s1)),1,num_levels^2);
        glcm_5 = reshape(GLCM(img,offset,2*offset,num_levels,sqrt(s1)),1,num_levels^2);

        %use this dimenstionality instead
        glcm(:,i) = [glcm_1,glcm_2,glcm_3,glcm_4,glcm_5]';
    
    end
    
    glcm = glcm / sum(glcm,'all');
    
end