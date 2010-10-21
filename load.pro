pro load, i

  common func, n1, n2, f

  name = string(i, format='(i04)') + '.raw'
  print, 'loading: ' + name

  openr, lun, name, /get_lun

    ; load array information
    n = lonarr(3)
    readu, lun, n
    n1 = n[1]
    n2 = n[2]

    ; load vorticity
    if n[0] eq 8 then f = dblarr(n2, n1) $
    else              f = fltarr(n2, n1)
    readu, lun, f
    f = transpose(f)

  close, lun & free_lun, lun

end
