function y = project_kda(a,D,X,kernel,alpha)
    p = length(X(1,:));
    y = zeros(1,p);
    for i = 1:p
        tot = 0;
        for j = 1:length(a)
            if strcmp(kernel,'rbf')
                tot = tot + a(j)*exp(alpha*vecnorm(D(:,j) - X(:,i))^2);
            else
                tot = tot + a(j)*(D(:,j)'*X(:,i))^alpha;
            end
        end
        y(i) = tot; 
    end
end