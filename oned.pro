function oned, k, E, n, cumul=cumul

  if not keyword_set(cumul) then cumul=0

  Ec = fltarr(n)
  kc = fltarr(n)
  nb = lonarr(n)
  kb = exp(alog(min(size(k, /dimensions)) / 2 + 0.99) * findgen(n + 1) / n)

  for i = 0, n-1 do begin

         if i eq 0   then j = where(0.0   lt k and k lt kb[i+1], count) $
    else if i eq n-1 then j = where(kb[i] le k and k le kb[i+1], count) $
    else                  j = where(kb[i] le k and k lt kb[i+1], count)

    nb[i] = count

    if count ne 0 then begin
      kc[i] = total(k[j]) / count
      Ec[i] = total(E[j])
    endif else begin
      kc[i] = sqrt(kb[i] * kb[i+1])
      Ec[i] = 0.0
    endelse

    if cumul and i gt 0 then Ec[i] = Ec[i] + Ec[i-1]

  endfor

  return, {E:Ec, k:kc, n:nb, b:kb}

end
