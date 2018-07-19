# Geographical Latent Representation for Predicting Future Visitors #

## main work ##

- a latent represent model, they use this model incorporates the geographical influence of POIs.In fact,they use word2vec to realize it. Their work is getting the probability of a user $u$ visiting a POI $l$ considering of the previous POI $C(l)$ the user $u$ visited.It is defined with a SoftMax function:$$Pr(l|C(l)) = e^{(w(l)\cdot\Phi(C(l)) )}/Z(C(l))$$

  Here,$w(l)$ is the latent vector for POI $l$ and $\Phi(C(l)) = \sum_{l_{c}\in C(l)}w(l_{c})$ means the sum of vector of contextual POIS.

  

---

## Question ##

1. What is POI? How we realize it in the code?

   POI is Point of Interest. It means a place.

2. 

