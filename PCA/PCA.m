%% Question 1
%plot of 3d data
load('DataAssignment1S2022.mat')
figure(1)
plot3(X(1,:),X(2,:),X(3,:),'k.','MarkerSize',5)
grid on
axis equal
xlabel 'row1'
ylabel 'row2'
zlabel 'row3'

%create plot of projected data
figure(2)
subplot(1,3,1)
plot(X(1,:),X(2,:),'k.','Markersize',5)
axis equal
xlabel 'Row 1'
ylabel 'Row 2'
subplot(1,3,2)
plot(X(2,:),X(3,:),'k.','Markersize',5)
axis equal
xlabel 'Row 2'
ylabel 'Row 3'
subplot(1,3,3)
plot(X(1,:),X(3,:),'k.','Markersize',5)
axis equal
xlabel 'Row 1'
ylabel 'Row 3'


%get the mean of each feature
xbar = mean(X,2);

%center the data
Xc = X - xbar;

%get the SVD of centered and uncentered data
[U,S,V] = svd(X);
[Uc,Sc,Vc] = svd(Xc);

%retrieve singular values
singular_values = diag(S);
singular_valuesc = diag(Sc);

%calculate the principle components
Z = U'*X;
Zc = Uc'*Xc;

%plot each of the PCs against one another for each dataset
%centered data
tiledlayout(2,3)
nexttile
plot(Zc(1,:),Zc(2,:),'k.','MarkerSize',5);
axis equal
xlabel 'PC1'
ylabel 'PC2'

nexttile
plot(Zc(1,:),Zc(3,:),'k.','MarkerSize',5);
axis equal
xlabel 'PC1'
ylabel 'PC3'

nexttile
plot(Zc(2,:),Zc(3,:),'k.','MarkerSize',5);
axis equal
xlabel 'PC2'
ylabel 'PC3'

%uncentered data
nexttile
plot(Z(1,:),Z(2,:),'k.','MarkerSize',5);
axis equal
xlabel 'PC1'
ylabel 'PC2'

nexttile
plot(Z(1,:),Z(3,:),'k.','MarkerSize',5);
axis equal
xlabel 'PC1'
ylabel 'PC3'

nexttile
plot(Z(2,:),Z(3,:),'k.','MarkerSize',5);
axis equal
xlabel 'PC2'
ylabel 'PC3'

%% Question 2

iris_data = load('IrisDataAnnotated.mat');

%getting mean and centering data
xbar = mean(iris_data.X,2);
iris_data.X = iris_data.X - xbar;

%calulating SVD, getting singular values, calculating principle components
[U,S,V] = svd(iris_data.X);
singular_values = diag(S);
Z = U'*iris_data.X;

%finding index vectors off each type
first = find(iris_data.I == 1);
sec = find(iris_data.I == 2);
third = find(iris_data.I == 3);

%plotting the first two principle components with color identification
figure(9)
plot(Z(1,first),Z(2,first),'b.','Markersize',15)
hold on
plot(Z(1,sec),Z(2,sec),'g.','Markersize',15)
hold on
plot(Z(1,third),Z(2,third),'r.','Markersize',15)
xlabel 'PC1'
ylabel 'PC2'
legend('Iris Setosa','Iris Versicolor','Iris Virginica','Fontsize',15)
axis equal

figure(10)
histogram(Z(1,first),'FaceColor','b','FaceAlpha',0.5)
hold on
histogram(Z(1,sec),'FaceColor','g','FaceAlpha',0.5)
hold on
histogram(Z(1,third),'FaceColor','r','FaceAlpha',0.5)
legend('Iris Setosa','Iris Versicolor','Iris Virginica','Fontsize',15)
xlabel 'PC1 Value'
ylabel 'Frequency'

%% Question 3

load('HandwrittenDigits.mat');

%get the index vectors of the corresponding digits
zeros_ind = find(I == 0);
fives_ind = find(I == 5);
eights_ind = find(I == 8);

%getting the images corresponding to the digits
zeros = X(:,zeros_ind);
fives = X(:,fives_ind);
eights = X(:,eights_ind);

%Computing SVD without centering
[U0,S0,V0] = svd(zeros);
[U5,S5,V5] = svd(fives);
[U8,S8,V8] = svd(eights);

figure(1);
%taking the first five feature vectors for each digit and creating images of them
tiledlayout(3,5,'TileSpacing','none')
for i = (1:5)
    nexttile(i)
    imagesc(reshape(U0(:,i),16,16)');
    colormap(gca,1-gray);
    axis square
    axis off
    title(strcat(num2str(i),' vectors'))

    nexttile(i+5)
    imagesc(reshape(U5(:,i),16,16)');
    colormap(gca,gray)
    axis square
    axis off
    title(strcat(num2str(i),' vectors'))
    
    nexttile(i+10)
    imagesc(reshape(U8(:,i),16,16)');
    colormap(gca,gray)
    axis square
    axis off
    title(strcat(num2str(i),' vectors'))
end


%getting representative
zero = zeros(:,1);
five = fives(:,1);
eight = eights(:,1);

%graphing the projected approximations
figure(2)
tiledlayout(3,5,"TileSpacing","tight")
zero_approx = create_approx(U0,zero);
five_approx = create_approx(U5,five);
eight_approx = create_approx(U8,eight);

%graphing the errors in the residuals of the approximations
figure(3)
tiledlayout(1,3,"TileSpacing","tight")
nexttile
bar(zero_approx)
xticklabels({'5','10','15','20','25'})
xlabel 'k value'
title 'Error for 0'
nexttile
bar(five_approx)
xticklabels({'5','10','15','20','25'})
title 'Error for 5'
xlabel 'k value'
nexttile
bar(eight_approx)
xticklabels({'5','10','15','20','25'})
title 'Error for 8'
xlabel 'k value'

%graphing the residuals
figure(4)
tiledlayout(1,3,"TileSpacing","none")
Pk = U0(:,1:25)*(U0(:,1:25)');
nexttile
imagesc(reshape(zero - Pk*zero,16,16)')
colormap(1-gray)
axis square
axis off
subtitle 'k = 25'
title ('Zero Residual')

Pk = U5(:,1:25)*(U5(:,1:25)');
nexttile
imagesc(reshape(five - Pk*five,16,16)')
colormap(1-gray)
axis square
axis off
subtitle 'k = 25'
title('Five Residual')

Pk = U8(:,1:25)*(U8(:,1:25)');
nexttile
imagesc(reshape(eight - Pk*eight,16,16)')
colormap(1-gray)
axis square
axis off
subtitle 'k = 25'
title ('Eight Residual')

%generating images of the reference used during approximation
figure(5)
tiledlayout(1,3,"TileSpacing","none")
nexttile
colormap(1-gray)
imagesc(reshape(zero,16,16)')
axis square
axis off
nexttile
imagesc(reshape(five,16,16)')
axis square
axis off
colormap(1-gray)
nexttile
imagesc(reshape(eight,16,16)')
axis square
axis off
colormap(1-gray)

%this function creates a projector matrix with 5,10,15,20 and 25 feature
%vectors and then creates an approximation image. It returns the error and
%produces graphs
%mat is a matrix that will be used to create the projector
%rep is a known image
%num is the number that is being looked at
function error = create_approx(mat,rep)
    err = (1:5);
    for i = (1:5)
        %projector matrix
        P = mat(:,1:5*i)*(mat(:,1:5*i)');

        %approximation
        approx = P*rep;

        %create image
        nexttile
        imagesc(reshape(approx,16,16)');
        colormap(1-gray)
        tit = strcat('k = ',num2str(5*i));
        title(tit)
        axis off

        %store error
        err(i) = norm(rep - approx);
    end
    error = err;
end

