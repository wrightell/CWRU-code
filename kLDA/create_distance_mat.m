function D = create_distance_mat(X,medoids)
    n = length(X);
    D = nan(n);
    for i = 1:n
        for j = 1:n
            %if symmetric point is already filled in, copy it
            if(not(isnan(D(j,i))))
                D(i,j) = D(j,i);
            else
                if medoids
                    D(i,j) = medoid_dist(X(:,i),X(:,j));
                else
                    D(i,j) = norm(X(:,i)-X(:,j));
                end
            end
        end
    end
end