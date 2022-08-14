load('WineData.mat')

%calculating directions and getting reduced matrix
Q = LDA(X,I);
Z = Q'*(X- mean(X,2));

%finding indices associated to each class
one = find(I == 1);
two = find(I == 2);
three = find(I == 3);

%plotting the reduced data
hold on
plot(Z(1,one),Z(2,one),'r.','MarkerSize',15)
plot(Z(1,two),Z(2,two),'g.','MarkerSize',15)
plot(Z(1,three),Z(2,three),'b.','MarkerSize',15)
xlabel 'Component 1'
ylabel 'Component 2'
title 'LDA Projection'

%getting the PCA of the data
Xc = X - mean(X,2);
[U,S,V] = svd(Xc);
ZZ = U'*Xc;

%plotting the PCs
figure(2)
title 'PCA Projection'
xlabel PC1
ylabel PC2
hold on
plot(ZZ(1,one),ZZ(2,one),'r.','MarkerSize',15)
plot(ZZ(1,two),ZZ(2,two),'g.','MarkerSize',15)
plot(ZZ(1,three),ZZ(2,three),'b.','MarkerSize',15)
axis square

figure(3)
title 'PCA Projection'
xlabel PC1
ylabel PC3
hold on
plot(ZZ(1,one),ZZ(3,one),'r.','MarkerSize',15)
plot(ZZ(1,two),ZZ(3,two),'g.','MarkerSize',15)
plot(ZZ(1,three),ZZ(3,three),'b.','MarkerSize',15)
axis square

figure(4)
title 'PCA Projection'
xlabel PC2
ylabel PC3
hold on
plot(ZZ(2,one),ZZ(3,one),'r.','MarkerSize',15)
plot(ZZ(2,two),ZZ(3,two),'g.','MarkerSize',15)
plot(ZZ(2,three),ZZ(3,three),'b.','MarkerSize',15)
axis square

