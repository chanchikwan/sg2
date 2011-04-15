function load, pre, num, quiet=quiet, odd=odd, check=check, view=view

  if n_elements(num) eq 0 then begin
    p = ''
    i = pre
  endif else begin
    p = pre
    i = num
  endelse
  if not keyword_set(quiet) then quiet = 0
  if not keyword_set(odd  ) then odd   = 0
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
    readu, lun, h

  close, lun & free_lun, lun

  ; constructe the full spectral array
  u = reverse([[h[1:*,0]], [reverse(h[1:*,1:*],2)]])
  s = [h[0:h2-2+odd,*], conj(u)]

  if check then begin
    f  = fft(s, /inverse, /double)
    Re = real_part(f)
    Im = imaginary(f)
    print, 'max(abs(Im)) / max(abs(Re)) = ', max(abs(Im)) / max(abs(Re))
  endif
  if view then tvscl, (2 * alog10(abs(s))) > (-16)

  return, s

end
