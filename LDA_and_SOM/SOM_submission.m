
%set the parameters
lattice = [1,50];
list = [1,30];

params1 = [.9, .01; max(list)/3, .5];
params2 = [.9, .01; max(lattice)/3, .5];

%create sine and circle data
t = 2*pi*[1:1000]/1000;

sinus = [t;sin(t)];
circle = [cos(t);sin(t)];

%add some noise
sinus(2,:) = sinus(2,:) + normrnd(0,.05,[1,1000]);
circle(2,:) = circle(2,:) + normrnd(0,.05,[1,1000]);

%calc approximations
m1 = SOM(sinus,list,params1,2000);
m2 = SOM(circle,lattice,params2,2000);

%plot the figures
figure(1)
plot(sinus(1,:),sinus(2,:),'k.','MarkerSize',10)
hold on
plot(m1(1,:),m1(2,:),'r.','MarkerSize',15)

figure(2)
plot(circle(1,:),circle(2,:),'k.','MarkerSize',10)
hold on
plot(m2(1,:),m2(2,:),'r.','MarkerSize',15)
axis equal
%% number 2
load('HandwrittenDigits.mat')
params = [.9, .01; 10/3, .5];

%get indices of 2,5,7
two = find(I == 2);
seven = find(I == 7);
five = find(I == 5);

%subset the data
sub = X(:,[two,five,seven]);

%get prototypes 
M = SOM(sub,[10,10],params,1500);
V = zeros(256,100);
for i = 1:100
    [~,best] = min(vecnorm(X - M(:,i)));
    V(:,i) = X(:,best);
end

%generate plot of comparison
tiledlayout(2,5,'TileSpacing','none')
for j = 1:5
    k = randi(100);
    nexttile(j)
    imagesc(reshape(V(:,k),16,16)')
    axis square
    axis off
    nexttile(j+5)
    imagesc(reshape(M(:,k),16,16)')
    axis square
    axis off
    colormap(1-gray);
end

%view the 2d map
tiledlayout(10,10,'Padding','compact','TileSpacing','none')
for i = 1:100
    nexttile
    imagesc(reshape(M(:,i),16,16)')
end
    
    
