%load and format parameters
load("ECG_test.mat")
load("ECG_train.mat")
X_train_abnormal = X_train_abnormal';
X_train_normal = X_train_normal';
X_test_abnormal = X_test_abnormal';
X_test_normal = X_test_normal';
X = [X_train_normal,X_train_abnormal];
I = [ones(1,length(X_train_normal(1,:))),2*ones(1,length(X_train_abnormal(1,:)))];
predicted = zeros(1,100)-1;

%run prediction with multiple k

%k = 1
slice = length(X_test_normal(1,:));
for i = 1:slice
    predicted(i) = KNN(X,I,1,X_test_normal(:,i));
end

for i = 1:length(X_test_abnormal(1,:))
    predicted(i+slice) = KNN(X,I,1,X_test_abnormal(:,i));
end

conf_1 = create_conf_mat(predicted-1,I-1);

%k = 3
for i = 1:slice
    predicted(i) = KNN(X,I,3,X_test_normal(:,i));
end

for i = 1:length(X_test_abnormal(1,:))
    predicted(i+slice) = KNN(X,I,3,X_test_abnormal(:,i));
end
conf_3 = create_conf_mat(predicted-1,I-1);

%k = 5
for i = 1:slice
    predicted(i) = KNN(X,I,5,X_test_normal(:,i));
end

for i = 1:length(X_test_abnormal(1,:))
    predicted(i+slice) = KNN(X,I,5,X_test_abnormal(:,i));
end
conf_5 = create_conf_mat(predicted-1,I-1);

%k = 7
for i = 1:slice
    predicted(i) = KNN(X,I,7,X_test_normal(:,i));
end

for i = 1:length(X_test_abnormal(1,:))
    predicted(i+slice) = KNN(X,I,7,X_test_abnormal(:,i));
end
conf_7 = create_conf_mat(predicted-1,I-1);


