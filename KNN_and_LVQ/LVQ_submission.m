
%load and format parameters
load("ECG_test.mat")
load("ECG_train.mat")
params = [.9, .01; 10/3, .5];
X_train_abnormal = X_train_abnormal';
X_train_normal = X_train_normal';
X_test_abnormal = X_test_abnormal';
X_test_normal = X_test_normal';
X = [X_train_normal,X_train_abnormal];
I = [ones(1,length(X_train_normal(1,:))),2*ones(1,length(X_train_abnormal(1,:)))];

%run algorithm
[M,Is] = LVQ(X,I,[1,15],params);


slice = length(X_test_normal(1,:));
predicted = zeros(1,length(I));

%classify the tests
for i = 1:slice
    [~,index] = min(vecnorm(M - X_test_normal(:,i)));
    predicted(i) = Is(index);
end

for i = 1:length(X_test_abnormal(1,:))
    [~,index] = min(vecnorm(M - X_test_abnormal(:,i)));
    predicted(i + slice) = Is(index);
end

%create confusion matrix
conf = create_conf_mat(predicted-1,I-1);


