function a = kLDA(X,I,kernel,alpha)
    
    %format the dataset and get the centers of the clusters
    [data,~,~,~] = process_data(X,I);
    
    if kernel == "rbf"
        K = exp(alpha*create_distance_mat(data,false).^2);
    else
        K = linear_k(X).^alpha;
    end

    p = length(I);

    M1 = zeros(p,1);
    M2 = zeros(p,1);
 

    for i = 1:p
        M1(i) = sum(K(i,I == -1));
        M2(i) = sum(K(i,I == 1));
    end

    K1 = K(:,I == -1);
    K2 = K(:,I == 1);

    p1 = sum(I == -1);
    p2 = sum(I == 1);

    N = K1*(eye(p1)- ones(p1)/p1)*K1' + K2*(eye(p2)-ones(p2)/p2)*K2';

    eigen = eigs(N,1);
    error = eigen*(10^-10);
    N = N + error*eye(p);

    a = N\(M2 - M1);

end




