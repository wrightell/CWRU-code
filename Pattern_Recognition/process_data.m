function [data,labels,centers,k] = process_data(X,I)
    
    %get all the classes
    classes = unique(I);
    k = length(classes);

    %store the sorted data and labels
    data = zeros(size(X));
    labels = zeros(1,length(data));
    
    %store the k centers
    n = length(X(:,1));
    centers = zeros(n,k);

    %keep track of the current index of the submatrix
    current_length = 1;

    for i = 1:k
        
        %get the cluster
        indices = (I == classes(i));
        cluster = X(:,indices);

        %calculate the mean
        centers(:,i) = mean(cluster,2);

        %count the size of this submatrix
        num_samples = sum(indices);
        final_length = current_length + num_samples-1;

        %put the data in order and center it, update the labels
        data(:,current_length:final_length) = cluster - centers(:,i);
        labels(current_length:final_length) = i;

        %update the new starting index
        current_length = final_length + 1; 
    end
end