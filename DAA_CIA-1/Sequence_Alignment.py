import numpy as np

def get_minimum_penalty(x:str, y:str, pxy:int, pgap:int):

	
	i = 0
	j = 0
	
	
	m = len(x)
	n = len(y)
	
	
	dp = np.zeros([m+1,n+1], dtype=int) 

	
	dp[0:(m+1),0] = [ i * pgap for i in range(m+1)]
	dp[0,0:(n+1)] = [ i * pgap for i in range(n+1)]

	
	i = 1
	while i <= m:
		j = 1
		while j <= n:
			if x[i - 1] == y[j - 1]:
				dp[i][j] = dp[i - 1][j - 1]
			else:
				dp[i][j] = min(dp[i - 1][j - 1] + pxy,
								dp[i - 1][j] + pgap,
								dp[i][j - 1] + pgap)
			j += 1
		i += 1
	
	
	l = n + m 
	i = m
	j = n
	
	xpos = l
	ypos = l

	
	xans = np.zeros(l+1, dtype=int)
	yans = np.zeros(l+1, dtype=int)
	

	while not (i == 0 or j == 0):
		
		if x[i - 1] == y[j - 1]:	
			xans[xpos] = ord(x[i - 1])
			yans[ypos] = ord(y[j - 1])
			xpos -= 1
			ypos -= 1
			i -= 1
			j -= 1
		elif (dp[i - 1][j - 1] + pxy) == dp[i][j]:
		
			xans[xpos] = ord(x[i - 1])
			yans[ypos] = ord(y[j - 1])
			xpos -= 1
			ypos -= 1
			i -= 1
			j -= 1
		
		elif (dp[i - 1][j] + pgap) == dp[i][j]:
			xans[xpos] = ord(x[i - 1])
			yans[ypos] = ord('_')
			xpos -= 1
			ypos -= 1
			i -= 1
		
		elif (dp[i][j - 1] + pgap) == dp[i][j]:	
			xans[xpos] = ord('_')
			yans[ypos] = ord(y[j - 1])
			xpos -= 1
			ypos -= 1
			j -= 1
		

	while xpos > 0:
		if i > 0:
			i -= 1
			xans[xpos] = ord(x[i])
			xpos -= 1
		else:
			xans[xpos] = ord('_')
			xpos -= 1
	
	while ypos > 0:
		if j > 0:
			j -= 1
			yans[ypos] = ord(y[j])
			ypos -= 1
		else:
			yans[ypos] = ord('_')
			ypos -= 1

	
	id = 1
	i = l
	while i >= 1:
		if (chr(yans[i]) == '_') and chr(xans[i]) == '_':
			id = i + 1
			break
		
		i -= 1

	
	print(f"Minimum Penalty in aligning the genes = {dp[m][n]}")
	print("The aligned genes are:")
	
	i = id
	x_seq = ""
	while i <= l:
		x_seq += chr(xans[i])
		i += 1
	print(f"X seq: {x_seq}")

	
	i = id
	y_seq = ""
	while i <= l:
		y_seq += chr(yans[i])
		i += 1
	print(f"Y seq: {y_seq}")

def test_get_minimum_penalty():
	"""
	Test the get_minimum_penalty function
	"""
	
	gene1 = "AGGGCT"
	gene2 = "AGGCA"
	
	
	mismatch_penalty = 3
	gap_penalty = 2

	
	get_minimum_penalty(gene1, gene2, mismatch_penalty, gap_penalty)

test_get_minimum_penalty()
