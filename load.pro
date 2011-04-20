function load, p, i, quiet=quiet, nonl=nonl, show=show

  if n_elements(i) eq 0 then name =     string(p, format='(i04)') $
  else                       name = p + string(i, format='(i04)')

  if not keyword_set(quiet) then print, 'loading: ' + name + '.raw'
  if not keyword_set(nonl ) then nonl  = 0
  if not keyword_set(show ) then show  = 0

  openr, lun, name + '.raw', /get_lun

    ; load array information
    n  = lonarr(4) & readu, lun, n
    n1 = n[1] & h2 = n[2] & n2 = 2 * h2 - 1
    if n[0] eq -8 then h =  complexarr(h2, n1) $
    else               h = dcomplexarr(h2, n1)

    ; constructe the full vorticity
    begin
      readu, lun, h
      u = reverse([[h[1:*,0]], [reverse(h[1:*,1:*],2)]])
      W = [h[0:h2-1,*], conj(u)]
      if show then tvscl, (2 * alog10(abs(W))) > (-16)
    endif

    ; constructe the full non-linear term
    if nonl then begin
      readu, lun, h
      u = reverse([[h[1:*,0]], [reverse(h[1:*,1:*],2)]])
      J = [h[0:h2-1,*], conj(u)]
      if show then tvscl, (2 * alog10(abs(J))) > (-16)
    endif

  close, lun & free_lun, lun

  if nonl then return, {name:name, W:W, J:J} $
  else         return, {name:name, W:W}

end
