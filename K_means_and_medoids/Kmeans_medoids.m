
%% Question 1
load("WineData.mat")
X = (X - mean(X,2));

[classes,centers,list] = k_means(X,3,.01,20);

[U,S,V] = svd(X);
Z = U'*X;
z11 = Z(1,classes == 1);
z21 = Z(2,classes == 1);
z12 = Z(1,classes == 2);
z22 = Z(2,classes == 2);
z13 = Z(1,classes == 3);
z23 = Z(2,classes == 3);
z11b = Z(1,I == 1);
z21b = Z(2,I == 1);
z12b = Z(1,I == 2);
z22b = Z(2,I == 2);
z13b = Z(1,I == 3);
z23b = Z(2,I == 3);

A = [sum(I==1),sum(I==2),sum(I==3);sum(classes==1),sum(classes==2),sum(classes==3)];
A

%plot(list)
figure(1)
plot(z11,z21,'g.','MarkerSize',15)
hold on
plot(z12,z22,'r.','MarkerSize',15)
hold on
plot(z13,z23,'b.','MarkerSize',15)
hold on
%figure(1)
%plot(z11b,z21b,'r.','MarkerSize',15)
%hold on
%plot(z12b,z22b,'g.','MarkerSize',15)
%hold on
%plot(z13b,z23b,'b.','MarkerSize',15)
legend('1','2','3')
axis square
xlabel 'PC1'
ylabel 'PC2'
%% Qestion 2

load("WineData.mat")
X = (X - mean(X,2));
[classes,centers,list] = k_medoids(X,3,.01,20,false);

[U,S,V] = svd(X);
Z = U'*X;
z11 = Z(1,classes == 1);
z21 = Z(2,classes == 1);
z12 = Z(1,classes == 2);
z22 = Z(2,classes == 2);
z13 = Z(1,classes == 3);
z23 = Z(2,classes == 3);
z11b = Z(1,I == 1);
z21b = Z(2,I == 1);
z12b = Z(1,I == 2);
z22b = Z(2,I == 2);
z13b = Z(1,I == 3);
z23b = Z(2,I == 3);

A = [sum(I==1),sum(I==2),sum(I==3);sum(classes==1),sum(classes==2),sum(classes==3)];
A

%plot(list)
figure(1)
plot(z11,z21,'b.','MarkerSize',15)
hold on
plot(z12,z22,'r.','MarkerSize',15)
hold on
plot(z13,z23,'g.','MarkerSize',15)
hold on
axis square
xlabel PC1
ylabel PC2
figure(2)
plot(z11b,z21b,'c.','MarkerSize',15)
hold on
plot(z12b,z22b,'m.','MarkerSize',15)
hold on
plot(z13b,z23b,'k.','MarkerSize',15)
%legend('pred_1','pred_2','pred_3','act_1','act_2','act_3')
axis square

%% Question 4

load("CongressionalVoteData.mat")
X(:,249) = [];
I(249) = [];
[classes,centers,list] = k_medoids(X,2,.01,20,true);

%calculate the confusion matrox
TP = sum((classes == 1) .* (I == 0));
FP = sum((classes == 1) .* (I == 1));
FN = sum((classes == 2) .* (I == 0));
TN = sum((classes == 2) .* (I == 1));
C = [TP, FP; FN, TN]








