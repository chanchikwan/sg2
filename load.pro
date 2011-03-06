pro load, i, surf=surf, view=view

  common func, n1, n2, f
  common spec, k1, k2, s

  if not keyword_set(surf) then surf = 0
  if not keyword_set(view) then view = 0

  name = string(i, format='(i04)') + '.raw'
  print, 'loading: ' + name

  openr, lun, name, /get_lun

    ; load array information
    n = lonarr(4)
    readu, lun, n
    n1 = n[1]
    n2 = n[2]

    ; load vorticity
    if n[0] eq 8 then f = dblarr(n2, n1) $
    else              f = fltarr(n2, n1)
    readu, lun, f
    f = transpose(f)

  close, lun & free_lun, lun

  ; construct k-grid
  k1 = [dindgen(n1-n1/2), -reverse(dindgen(n1/2)+1)]
  k2 = [dindgen(n2-n2/2), -reverse(dindgen(n2/2)+1)]
  k1 =           rebin(k1, n1, n2)
  k2 = transpose(rebin(k2, n2, n1))

  ; obtain fft
  s = fft(f)
  if surf then shade_surf, 2 * alog10(abs(s)) > (-16)
  if view then tvscl, 2 * alog10(abs(s)) > (-16)

end
