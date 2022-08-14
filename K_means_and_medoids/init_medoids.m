function [labels,coherence,centers] = init_medoids(n,D,k)
    
    p = length(D);
    labels = [];
    coherence = Inf;
    for i = 1:n

        %pick k random points as the center
        medoids = randsample(p,k);
        d_bar = D(medoids,:);

        %calculate the within cluster coherence while assigning vectors to
        %min medoid
        [dist,pointers] = min(d_bar);
        Q = sum(dist);

        %only update if better
        if Q < coherence
            coherence = Q;
            labels = pointers;
            centers = medoids;
        end
    end
    
end
