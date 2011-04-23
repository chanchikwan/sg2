function oned, Z, n

  kk = getkk(Z)
  E  = Z / kk & E[0] = 0.0
  k  = sqrt(kk)

  kb = exp(alog(min(size(k, /dimensions)) / 2 + 0.99) * findgen(n + 1) / n)
  nb = lonarr(n)
  kc = sqrt(kb[0:n-1] * kb[1:n])
  Zc = fltarr(n)
  Ec = fltarr(n)

  for i = 0, n-1 do begin

    case i of
      0    : j = where(0.0   lt k and k lt kb[i+1], count)
      n-1  : j = where(kb[i] le k and k le kb[i+1], count)
      else : j = where(kb[i] le k and k lt kb[i+1], count)
    endcase

    if count ne 0 then begin
      Ec[i] = total(E[j])
      Zc[i] = total(Z[j])
      kc[i] = total(k[j]) / count
      nb[i] = count
    endif

  endfor

  return, {E:Ec, Z:Zc, k:kc, n:nb, b:kb}

end
