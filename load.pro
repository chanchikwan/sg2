function load, pre, num, quiet=quiet, even=even, check=check, view=view

  if n_elements(num) eq 0 then begin
    p = ''
    i = pre
  endif else begin
    p = pre
    i = num
  endelse
  if not keyword_set(quiet) then quiet = 0
  if not keyword_set(even ) then even  = 0
  if not keyword_set(check) then check = 0
  if not keyword_set(view ) then view  = 0

  name = p + string(i, format='(i04)') + '.raw'
  if not quiet then print, 'loading: ' + name

  openr, lun, name, /get_lun

    ; load array information
    n = lonarr(4)
    readu, lun, n
    n1 = n[1]
    h2 = n[2]

    ; load vorticity
    if n[0] eq -8 then h =  complexarr(h2, n1) $
    else               h = dcomplexarr(h2, n1)

    ; constructe the full vorticity
    readu, lun, h
    u = reverse([[h[1:*,0]], [reverse(h[1:*,1:*],2)]])
    W = [h[0:h2-1-even,*], conj(u)]
    if view then tvscl, (2 * alog10(abs(W))) > (-16)

    ; constructe the full non-linear term
    readu, lun, h
    u = reverse([[h[1:*,0]], [reverse(h[1:*,1:*],2)]])
    J = [h[0:h2-1-even,*], conj(u)]
    if view then tvscl, (2 * alog10(abs(J))) > (-16)

  close, lun & free_lun, lun

  if check then begin
    f = fft(W, /inverse, /double)
    print, 'max[abs(Im W)] / max[abs(Re W)] = ', max(abs(imaginary(f))) $
                                               / max(abs(real_part(f)))
    f = fft(J, /inverse, /double)
    print, 'max[abs(Im J)] / max[abs(Re J)] = ', max(abs(imaginary(f))) $
                                               / max(abs(real_part(f)))
  endif

  return, {W:W, J:J}

end
