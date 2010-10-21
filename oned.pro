function oned, k, E, bsz

  n  = ceil(alog(max(k)) / alog(bsz))
  kb = [0,bsz^(findgen(n+1))]
  kc = bsz^(findgen(n+1)-0.5)
  Ec = fltarr(n+1)

  for i = 0, n do begin
    j = where(kb[i] le k and k lt kb[i+1])
    Ec[i] = total(E[j]) / kc[i] ; divided by kc[i] to take away the log-bin
  endfor

  return, {k:kc, E:Ec}

end
