function getkk, W

  n  = size(W, /dimensions)
  n1 = n[0]
  n2 = n[1]

  kk1 = [findgen(n1-n1/2), -reverse(findgen(n1/2)+1)]^2
  kk2 = [findgen(n2-n2/2), -reverse(findgen(n2/2)+1)]^2

  return, rebin(kk1,n1,n2) + rebin(transpose(kk2),n2,n1)

end
