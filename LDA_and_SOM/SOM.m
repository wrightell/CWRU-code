
function prototypes = SOM(X,size,params,learning_time)

%initializing lattice and prototypes
lattice = create_lattice(size);
D = create_distance_mat(lattice,false);
K = size(1)*size(2);
T_max = 500*K;
p = length(X(1,:));
prototypes = X(:,randsample(p,K));

%this code is used for Question 3
%prototypes = zeros([256,K]);
%for i = 1:K
%  prototypes(:,i) = mean(X,2) + normrnd(0,.1,[1,256])';
%end

t = 1;

while t < T_max
    
    %pick a data point
    x = X(:,randi(p));

    %get index of best matching unit
    [~,BMU] = min(vecnorm(prototypes - x));
    
    %calculate the parameters
    alpha = max(params(1,1)*(1 - (t/learning_time)),params(1,2));
    gamma = max(params(2,1)*(1 - (t/learning_time)),params(2,2));
    
    %generate neighborhood matrix column for BMU
    h = exp((D(:,BMU).^2)/(-2*(gamma^2)));
    
    %update the prototypes
    for i = 1:K
        d = (x - prototypes(:,i));
        prototypes(:,i) = prototypes(:,i) + alpha*h(i)*d;
    end
    t = t + 1;
end
end