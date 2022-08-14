%load and format parameters
load("ECG_test.mat")
load("ECG_train.mat")
X_train_abnormal = X_train_abnormal';
X_train_normal = X_train_normal';
X_test_abnormal = X_test_abnormal';
X_test_normal = X_test_normal';
X = [X_train_normal,X_train_abnormal];
I = [ones(1,length(X_train_normal(1,:))),2*ones(1,length(X_train_abnormal(1,:)))];

%get best direction
q = LDA(X,I);

%project data
projected_data = q'*X;
normal = projected_data(projected_data < 4.7);
abnormal = projected_data(projected_data > 4.7);
histogram(normal,'FaceColor','r')
hold on
histogram(abnormal,'FaceColor','g')
legend('FontSize',20)

%do the same with test data
normal_test_projection = q'*X_test_normal;
abnormal_test_projection = q'*X_test_abnormal;
figure(2)
histogram(normal_test_projection,'FaceColor','r')
hold on
histogram(abnormal_test_projection,'FaceColor','g')
legend('normal','abnormal','FontSize',20)

figure(3)
histogram(q'*[X_test_normal,X_test_abnormal])