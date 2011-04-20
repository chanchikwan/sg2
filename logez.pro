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

    d  = load(p, i, /quiet)
    Zk = 0.5 * abs(d.W)^2 & Zk[0] = 0 ; the 2D enstrophy spectrum
    Ek = Zk / getkk(d.W)  & Ek[0] = 0 ; the 2D energy spectrum E(kx,ky)

    Z = total(Zk)
    E = total(Ek)
    print,       i, E, Z
    printf, lun, i, E, Z

  endfor
  close, lun & free_lun, lun

end
