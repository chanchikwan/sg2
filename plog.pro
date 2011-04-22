pro plog, n, Z=Z
  
  E = dblarr(n + 1)

  for i = 0, n do begin

    c = cache(string(i, format='(i04)') + '.sca')
    if keyword_set(Z) then E[i] = total(c.Z) $
    else                   E[i] = total(c.E)
     
  endfor

  plot, E

end
