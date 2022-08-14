
function [labels,centers,list] = k_medoids(data,k,tau,n,binary)
    
    %create the distance matrix
    D = create_distance_mat(data,binary);

    %initialize the medoids
    [labels,Q,centers] = init_medoids(n,D,k);
    
    %store the iteration number and  learning curve
    counter = 1;
    list = [];
    dQ = Inf;

    while dQ  > tau
        
        %update the centers
        centers = update_centers(D,labels,k);

        %update the labels and within-medoid coherences
        medoids = D(centers,:);
        [q,labels] = min(medoids);
        
        %calculate total coherence
        Q_new = sum(q);

        %calculate percent change in coherence
        dQ = abs(Q_new - Q)/Q;
        Q = Q_new;
        list(counter) = dQ;
        counter = counter + 1;
    end
end