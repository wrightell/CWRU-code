function [classes,centers,list] = k_means(data,k,tau,n)
   dQ = Inf;
   list = [];
   [characteristic_vecs,I_ell,Q] = init_clusters(data,k,n);
   counter = 1;
   while dQ > tau
        characteristic_vecs = update_means(k,data,I_ell);
        I_ell = update_assignment(characteristic_vecs,data);
        q1 = sum(within_cluster_coherence(data,I_ell,characteristic_vecs));
        dQ = abs(q1 - Q)/Q;
        Q = q1;
        list(counter) = dQ;
        counter = counter + 1;
   end
   classes = I_ell;
   centers = characteristic_vecs;
end