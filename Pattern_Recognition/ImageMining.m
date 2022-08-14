load('TestImages.mat')

%set parameters
offset = 5;
num_levels = 6;

%get principle component and linear discriminant data transformations
glcm = image_mine(X,I,num_levels,offset);

[U,~,~] = svd(glcm);
PC = U'*glcm;
PC4 = PC(1:4,:);

q = LDA(glcm,I);
LD = q'*glcm;
LD4 = LD(1:3,:);


t = LD4*PC4';

%plot them
plot_pcas(PC,I,offset,num_levels);
plot_lda(LD,I,offset,num_levels)


