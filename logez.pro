function getez, p, i, quiet=quiet

  common func, n1, n2, f
  common spec, k1, k2, s

  load, p, i, quiet=quiet

  kk = k1^2 + k2^2
  k  = sqrt(kk)
  Z  = abs(s)^2 & Z[0] = 0 ; the 2D enstrophy
  E  = Z / kk   & E[0] = 0 ; the 2D spectrum E(kx,ky)

  return, {E:0.5 * total(E), Z:0.5 * total(Z)}

end

pro logez, p, s, n, skip=skip

  if size(p,/type) eq 7 then begin ; p is string, must be the prefix
    if size(n,/type) eq 0 then begin
      if size(s,/type) eq 0 then n = 1024 else n = s
      s = 0
    endif
  end else begin
    if size(s,/type) eq 0 then begin
      if size(p,/type) eq 0 then n = 1024 else n = p
      s = 0
    endif else begin
      n = s
      s = p
    endelse
    p = ''
  endelse

  if not keyword_set(skip) then skip = 1

  openw, lun, p + 'log.txt', /get_lun
  for i = s, n, skip do begin
    c = getez(p, i, /quiet)
    print,       i, c.E, c.Z
    printf, lun, i, c.E, c.Z
  endfor
  close, lun & free_lun, lun

end
