# Jarvis march
import cv2

def jarvismarch(A):
  n = len(A)
  P = range(n)
  # start point
  for i in range(1,n):
    if A[P[i]][0]<A[P[0]][0]:
      P[i], P[0] = P[0], P[i]
  H = [P[0]]
  del P[0]
  P.append(H[0])
  while True:
    right = 0
    for i in range(1,len(P)):
      if rotate(A[H[-1]],A[P[right]],A[P[i]])<0:
        right = i
    if P[right]==H[0]:
      break
    else:
      H.append(P[right])
      del P[right]
  return H

im = cv2.imread("./test.png")
jarvismarch(im)
