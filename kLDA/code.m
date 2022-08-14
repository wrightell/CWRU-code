%%
% Generating dodgeball setup
%get x and y data
rng(5) %reproducability

%blue team
x1 = normrnd(2.5,.8,[1,100]);
y1 = unifrnd(0,5, [1,100]);

%red team
x2 = normrnd(7.5,.9,[1,100]);
y2 = unifrnd(0,5, [1,100]);

%create dataset and labels
X = [x1, x2; y1,y2];
I = [-1*ones(1,100),ones(1,100)];

%plotting initial setup
figure(1)
plot(x1,y1,'b.','MarkerSize',15)
hold on
plot(x2,y2,'r.','MarkerSize',15)

%creating svm model and getting support vectors
cl = fitcsvm(X',I);
svs = cl.SupportVectors;

%getting SVM margin
m = (svs(2,2) - svs(1,2))/(svs(2,1) - svs(1,1));
b = svs(2,2) - m*svs(2,1);
b2 = svs(4,2) - m*svs(4,1);

%plotting simple hyperplane
figure(2)
tiledlayout(1,2,"TileSpacing",'compact')
nexttile
plot(x1,y1,'b.','MarkerSize',15)
hold on
plot(x2,y2,'r.','MarkerSize',15)
xline(max(x1))
xline(min(x2))
axis square
title('Example Choice')

%plotting svm hyperplane
nexttile
plot(x1,y1,'b.','MarkerSize',15)
hold on
plot(x2,y2,'r.','MarkerSize',15)
axis square
refline(m,b)
refline(m,b2)
ylim([0 5])
plot(X(1,cl.IsSupportVector),X(2,cl.IsSupportVector),'ko','markersize',10)
title('SVM Choice')


%finding LDA direction
q = LDA(X,I);
q = q/vecnorm(q);

%projecting data
dball_proj = q'*X;

%plotting histogram
figure(3)
histogram(dball_proj(1:100),'NumBins',10,'facecolor','b')
hold on
histogram(dball_proj(101:200),'NumBins',10,'facecolor','r')
title("LDA Projection")

%%
clear
% Generating curved dodgeball setup

rng(1); % For reproducibility

%red team
radius = sqrt(rand(100,1));
theta = pi*rand(100,1);
data1 = [radius.*cos(theta), radius.*sin(theta)]; 

%blue team
radius2 = sqrt(3*rand(100,1)+1);
theta2 = pi*rand(100,1);
data2 = [radius2.*cos(theta2), radius2.*sin(theta2)];

%create training and testing set split 70-30%
train = [data1(1:70,:);data2(1:70,:)]';
test = [data1(71:100,:);data2(71:100,:)]';

%creating labels
I_train = [-1*ones(1,70),ones(1,70)];
I_test = [-1*ones(1,30),ones(1,30)];

%plot setup
figure(4)
plot(data1(:,1),data1(:,2),'r.','MarkerSize',15)
hold on
plot(data2(:,1),data2(:,2),'b.','MarkerSize',15)
th = linspace(0, pi, 100);
x = cos(th);
y = sin(th);
plot(x,y)
axis equal


%rbf svm
rbf_svm = fitcsvm(train',I_train,'KernelFunction','rbf');
rbf_svm_pred = rbf_svm.predict(test');
rbf_svm_acc = sum(rbf_svm_pred == I_test')/length(I_test);

poly_svm = fitcsvm(train',I_train,'KernelFunction','polynomial');
poly_svm_pred = poly_svm.predict(test');
poly_svm_acc = sum(poly_svm_pred == I_test')/length(I_test);

basic_svm = fitcsvm(train',I_train,'KernelFunction','linear');
basic_svm_pred = basic_svm.predict(test');
basic_svm_acc = sum(basic_svm_pred == I_test')/length(I_test);

%show support vectors
figure(5)
tiledlayout(3,1)
nexttile
plot(train(1,1:70),train(2,1:70),'r.','Markersize',15)
hold on
plot(train(1,71:140),train(2,71:140),'b.','Markersize',15)
plot(train(1,rbf_svm.IsSupportVector),train(2,rbf_svm.IsSupportVector),'ko','Markersize',10)
title("RBF SVM, Acc = " + string(rbf_svm_acc))

nexttile
plot(train(1,1:70),train(2,1:70),'r.','Markersize',15)
hold on
plot(train(1,71:140),train(2,71:140),'b.','Markersize',15)
plot(train(1,poly_svm.IsSupportVector),train(2,poly_svm.IsSupportVector),'ko','Markersize',10)
title("Polynomial SVM, Acc = " + string(poly_svm_acc))

nexttile
plot(train(1,1:70),train(2,1:70),'r.','Markersize',15)
hold on
plot(train(1,71:140),train(2,71:140),'b.','Markersize',15)
plot(train(1,basic_svm.IsSupportVector),train(2,basic_svm.IsSupportVector),'ko','Markersize',10)
title("Linear SVM, Acc = " + string(basic_svm_acc))


rbf_alpha = -7;
%create rbf-klda with best parameters
rbf_klda = kLDA(train,I_train,'rbf',rbf_alpha);

%tune for best split
p = project_kda(rbf_klda,train,train,'rbf',rbf_alpha);
split_rbf = find_best_split(p,I_train);

%get projection
rbf_klda_proj = project_kda(rbf_klda,train,test,'rbf',rbf_alpha);

%get predictions
rbf_klda_pred = sign(rbf_klda_proj - split_rbf);

%get accuracy
rbf_klda_acc = sum(rbf_klda_pred == I_test)/length(I_test);

%find misclassifications
rbf_klda_miss = find(rbf_klda_pred ~= I_test);

%plot histogram and mistakes
figure(6)
%tiledlayout(1,2)
tiledlayout(3,2)
nexttile
histogram(rbf_klda_proj(1:30),'numbins',10,'facecolor','r')
hold on
histogram(rbf_klda_proj(31:60),'numbins',10,'facecolor','b')
title("RBF-kLDA Projection")
xline(split_rbf,'linewidth',3)
axis square

nexttile
plot(test(1,1:30),test(2,1:30),'r.','MarkerSize',15)
hold on
plot(test(1,31:60),test(2,31:60),'b.','MarkerSize',15)
axis square
plot(test(1,rbf_klda_miss),test(2,rbf_klda_miss),'ko','MarkerSize',10)
title("RBF-kLDA Missclassifications, Acc = "+ string(rbf_klda_acc))


alpha_poly = 2;
%create poly-klda with best parameters 
poly_klda = kLDA(train,I_train,'polynomial',alpha_poly);

%tune for best split
p = project_kda(poly_klda,train,train,'polynomial',alpha_poly);
split_poly = find_best_split(p,I_train);

%get projection
poly_klda_proj = project_kda(poly_klda,train,test,'polynomial',alpha_poly);

%get predictions
poly_klda_pred = sign(poly_klda_proj - split_poly);

%get accuracy
poly_kdla_acc = sum(poly_klda_pred == I_test)/length(I_test);

%get misclassifications
poly_klda_miss = find(poly_klda_pred ~= I_test);

%plot histogram and misclassifications
%figure(7)
%tiledlayout(1,2)
nexttile
histogram(poly_klda_proj(1:30),'numbins',10,'facecolor','r')
hold on
histogram(poly_klda_proj(31:60),'numbins',10,'facecolor','b')
xline(split_poly,'linewidth',3)
title('Polynomial-kLDA Projection')
axis square

nexttile
plot(test(1,1:30),test(2,1:30),'r.','MarkerSize',15)
hold on
plot(test(1,31:60),test(2,31:60),'b.','MarkerSize',15)
axis square
plot(test(1,poly_klda_miss),test(2,poly_klda_miss),'ko','MarkerSize',10)
title("Polynomial-kLDA Missclassifications, Acc = "+string(poly_kdla_acc))

%create linear klda
basic_klda = kLDA(train,I_train,'linear',1);

%tune for best split
p = project_kda(basic_klda,train,train,'linear',1);
split_linear = find_best_split(p,I_train);

%get projections
basic_klda_proj = project_kda(basic_klda,train,test,'linear',1);

%get predictions
basic_klda_pred = sign(basic_klda_proj - split_linear);

%get accuracy
basic_kdla_acc = sum(basic_klda_pred == I_test)/length(I_test);

%get misscalssifications
basic_klda_miss = find(basic_klda_pred ~= I_test);

%plot histogram and misclassifications
%figure(8)
%tiledlayout(1,2)
nexttile
histogram(basic_klda_proj(1:30),'numbins',10,'facecolor','r')
hold on
histogram(basic_klda_proj(31:60),'numbins',10,'facecolor','b')
xline(split_linear,'linewidth',3)
title('Linear-kLDA Projection')
axis square

nexttile
plot(test(1,1:30),test(2,1:30),'r.','MarkerSize',15)
hold on
plot(test(1,31:60),test(2,31:60),'b.','MarkerSize',15)
axis square
plot(test(1,basic_klda_miss),test(2,basic_klda_miss),'ko','MarkerSize',10)
title("Linear-kLDA Missclassifications, Acc = "+string(basic_kdla_acc))


%%
% Transforming curved setup to 3d
X_3d = [train(1,:);train(2,:);train(1,:).^2 + train(2,:).^2];

figure(9)
plot3(X_3d(1,1:70),X_3d(2,1:70),X_3d(3,1:70),'r.','MarkerSize',15)
hold on
plot3(X_3d(1,71:140),X_3d(2,71:140),X_3d(3,71:140),'b.','MarkerSize',15)

%%
clear
% Formatting ECG data
load('ECG_train.mat')
data = [X_train_abnormal',X_train_normal'];
I = [-1*ones(1,31),ones(1,69)];

%model reduction with PCA remove for non pca
[U,D,V] = svd(data);
X = U(1:20,:)*D*V';

load('ECG_test.mat')
datat = [X_test_abnormal',X_test_normal'];

%remove this for nonPCA
[UU,DD,VV] = svd(datat);
Xt = UU(1:20,:)*DD*VV';

It = [-1*ones(1,36),ones(1,64)];

%rbf svm
rbf_svm = fitcsvm(X',I,'KernelFunction','rbf');
rbf_svm_pred = rbf_svm.predict(Xt');
rbf_svm_acc = sum(rbf_svm_pred == It')/length(It);

%poly svm
poly_svm = fitcsvm(X',I,'KernelFunction','polynomial');
poly_svm_pred = poly_svm.predict(Xt');
poly_svm_acc = sum(poly_svm_pred == It')/length(It);

%basic svm
linear_svm = fitcsvm(X',I,'KernelFunction','linear');
linear_svm_pred = linear_svm.predict(Xt');
linear_svm_acc = sum(linear_svm_pred == It')/length(It);


%rbf klda
%use -.05 for nonPCA, -.5 for PCA
rbf_alpha = -.5;
rbf_lda = kLDA(X,I,'rbf',rbf_alpha);

%tune for best split
p = project_kda(rbf_lda,X,X,'rbf',rbf_alpha);
rbf_split = find_best_split(p,I);

%get projections
rbf_lda_proj = project_kda(rbf_lda,X,Xt,'rbf',rbf_alpha);

%get predictions
rbf_lda_pred = sign(rbf_lda_proj - rbf_split);

%get accuracy
rbf_lda_acc = sum(rbf_lda_pred == It)/length(It);

%poly kdla
%use this for PCA, use 9 for non-PCA
poly_alpha = 2;
poly_lda = kLDA(X,I,'polynomial',poly_alpha);

%tune for best split
p = project_kda(poly_lda,X,X,'polynomial',poly_alpha);
poly_split = find_best_split(p,I);

%get projections
poly_lda_proj = project_kda(poly_lda,X,Xt,'polynomial',poly_alpha);

%get predictions
poly_lda_pred = sign(poly_lda_proj - poly_split);

%get accuracy
poly_lda_acc = sum(poly_lda_pred == It)/length(It);



%basic klda
linear_lda = kLDA(X,I,'linear',1);

%tune for best split
p = project_kda(linear_lda,X,X,'linear',1);
linear_split = find_best_split(p,I);

%get projections
linear_lda_proj = project_kda(linear_lda,X,Xt,'linear',1);

%get predictions
linear_lda_pred = sign(linear_lda_proj - linear_split);

%get accuracy
linear_lda_acc = sum(linear_lda_pred == It)/length(It);

%plot histograms
figure(10)
tiledlayout(1,3)
nexttile
histogram(poly_lda_proj(1:31),'NumBins',10)
hold on
histogram(poly_lda_proj(32:100),'NumBins',10)
title("Polynomial LDA, Acc = "+string(poly_lda_acc))
xline(poly_split,'linewidth',3)
legend('abnormal','normal')

nexttile
histogram(rbf_lda_proj(1:31),'NumBins',10)
hold on
histogram(rbf_lda_proj(32:100),'NumBins',10)
title("RBF LDA, Acc = "+string(rbf_lda_acc))
xline(rbf_split,'linewidth',3)
legend('abnormal','normal')

nexttile
histogram(linear_lda_proj(1:31),'NumBins',10)
hold on
histogram(linear_lda_proj(32:100),'NumBins',10)
title("Linear LDA, Acc = "+string(linear_lda_acc))
xline(linear_split,'linewidth',3)
legend('abnormal','normal')

tbl = 100*[rbf_svm_acc,poly_svm_acc,linear_svm_acc;rbf_lda_acc,poly_lda_acc,linear_lda_acc];