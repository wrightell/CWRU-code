function class = KNN(data,I,k,unknown)
    
    %get k indices of smallest distances
    [~,indices] = mink(vecnorm(data - unknown),k);

    %get the classes of the k-nearest neighbors
    neighbor_classes = I(indices);

    %majority vote wins
    class =  mode(neighbor_classes);
end
