%load and format parameters
load("ECG_test.mat")
load("ECG_train.mat")
X_train_abnormal = X_train_abnormal';
X_train_normal = X_train_normal';
X_test_abnormal = X_test_abnormal';
X_test_normal = X_test_normal';
X_train = [X_train_normal,X_train_abnormal];
train_labels = [ones(1,length(X_train_normal(1,:)))-1,ones(1,length(X_train_abnormal(1,:)))];
test_labels = [ones(1,length(X_test_normal(1,:)))-1,ones(1,length(X_test_abnormal(1,:)))];
X_test = [X_test_normal,X_test_abnormal];
knn_predictions = zeros(100,4);
lda_predictions = zeros(1,100);
knn_lvq_predictions = zeros(100,4);
t = length(X_test(1,:));

%getting lda direction
lda_direction = LDA(X_train,train_labels);

%getting prototypes
[lvq_prototypes,prototype_labels] = LVQ(X_train,train_labels,5);

%loop through each test point
for i = 1:t
 
    point = X_test(:,i);

    for k = [1,3,5,7]
        knn_predictions(i,(k+1)/2) = KNN(X_train,train_labels,k,point);
        knn_lvq_predictions(i,(k+1)/2) = KNN(lvq_prototypes,prototype_labels,k,point);
    end
end

knn_conf_1 = create_conf_mat(knn_predictions(:,1),test_labels);
knn_conf_3 = create_conf_mat(knn_predictions(:,2),test_labels);
knn_conf_5 = create_conf_mat(knn_predictions(:,3),test_labels);
knn_conf_7 = create_conf_mat(knn_predictions(:,4),test_labels);

lvq_conf_1 = create_conf_mat(knn_lvq_predictions(:,1),test_labels);
lvq_conf_3 = create_conf_mat(knn_lvq_predictions(:,2),test_labels);
lvq_conf_5 = create_conf_mat(knn_lvq_predictions(:,3),test_labels);
lvq_conf_7 = create_conf_mat(knn_lvq_predictions(:,4),test_labels);

figure(1)
histogram(lda_direction'*X_train_normal,'FaceColor','r')
title 'Training Set Projection'
hold on
histogram(lda_direction'*X_train_abnormal,'FaceColor','g')
legend('normal','abnormal','FontSize',20)

figure(2)
histogram(lda_direction'*X_test_normal,'FaceColor','r')
title 'Test Set Projection'
hold on
histogram(lda_direction'*X_test_abnormal,'FaceColor','g')
legend('normal','abnormal','FontSize',20)

figure(3)
tiledlayout(2,5,'TileSpacing','none')
for i = 1:10
    nexttile
    plot(lvq_prototypes(:,i))
    set(gca,'XTick',[], 'YTick', [])
end
