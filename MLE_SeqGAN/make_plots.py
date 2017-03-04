import matplotlib.pyplot as plt  
import cPickle 
import numpy as np
  
# Read the data into a pandas DataFrame.      
data = cPickle.load(open('save/mle-loss-20170304-015334.pkl'))

# These are the "Tableau 20" colors as RGB.    
tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),    
             (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),    
             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),    
             (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),    
             (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]    
  
# Scale the RGB values to the [0, 1] range, which is the format matplotlib accepts.    
for i in range(len(tableau20)):    
    r, g, b = tableau20[i]    
    tableau20[i] = (r / 255., g / 255., b / 255.)    
  
# You typically want your plot to be ~1.33x wider than tall. This plot is a rare    
# exception because of the number of lines being plotted on it.    
# Common sizes: (10, 7.5) and (12, 9)    
plt.figure(figsize=(12, 9))    
  
# Remove the plot frame lines. They are unnecessary chartjunk.    
ax = plt.subplot(111)    
ax.spines["top"].set_visible(False)    
ax.spines["bottom"].set_visible(False)    
ax.spines["right"].set_visible(False)    
ax.spines["left"].set_visible(False)    
  
# Ensure that the axis ticks only show up on the bottom and left of the plot.    
# Ticks on the right and top of the plot are generally unnecessary chartjunk.    
ax.get_xaxis().tick_bottom()    
ax.get_yaxis().tick_left()  
print data[0,0], data[-1,0]
xmin, xmax, ymin, ymax = data[0, 0], data[-1, 0], 8.5, 10.5  
  
# Limit the range of the plot to only where the data is.    
# Avoid unnecessary whitespace.    
plt.ylim(ymin, ymax)    
plt.xlim(xmin, xmax)    
  
# Make sure your axis ticks are large enough to be easily read.    
# You don't want your viewers squinting to read your plot.    
plt.yticks(fontsize=14)    
plt.xticks(fontsize=14)    
plt.xlabel('Epochs', fontsize=24)
plt.ylabel('NLL by oracle', fontsize=24)
plt.title('Learning curve', fontsize=30)
  
# Provide tick lines across the plot to help your viewers trace along    
# the axis ticks. Make sure that the lines are light and small so they    
# don't obscure the primary data lines.    
for y in np.linspace(ymin, ymax, 5):
    plt.plot(range(int(xmin), int(xmax)), [y] * len(range(int(xmin), int(xmax))), "--", lw=0.5, color="black", alpha=0.3)    
  
# Remove the tick marks; they are unnecessary with the tick lines we just plotted.    
plt.tick_params(axis="both", which="both", bottom="off", top="off",    
                labelbottom="on", left="off", right="off", labelleft="on")    
  
# Now that the plot is prepared, it's time to actually plot the data!    
# Note that I plotted the majors in order of the highest % in the final year.    
methods = ['MLE']  

for rank, column in enumerate(methods):    
    # Plot each line separately with its own color, using the Tableau 20    
    # color set in order.    
    plt.plot(data[:, 0], data[:,1],    
            lw=2.5, color=tableau20[rank])
  
    # Add a text label to the right end of every line. Most of the code below    
    # is adding specific offsets y position because some labels overlapped.    
    y_pos = data[-1,1]    
  
    # Again, make sure that all labels are large enough to be easily read    
    # by the viewer.    
    plt.text(xmax + 1.5, y_pos, column, fontsize=14, color=tableau20[rank])        
    
  
# Finally, save the figure as a PNG.    
# You can also save it as a PDF, JPEG, etc.    
# Just change the file extension in this call.    
# bbox_inches="tight" removes all the extra whitespace on the edges of your plot.    
plt.savefig("learning-rate.png", bbox_inches="tight")