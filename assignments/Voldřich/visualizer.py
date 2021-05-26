import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

class Visualizer:
    def __init__(self, W):
        self.W=W
    
    #rescales the lists for colormap usages
    def rescale(self,y):
        return (y - np.min(y)) / (np.max(y) - np.min(y))
    
    #generates cmap of hsv
    def getCmap(self,n, name='hsv'):
        '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
        RGB color; the keyword argument name must be a standard mpl colormap name.'''
        return plt.cm.get_cmap(name,n)
    
    #create simple bar chart with no axis and frame
    def barChart(self,x,y,ticklabels=None,xLabel='x axis',yLabel='y axis',title='Bar Chart'):
        #if no x specified, just use natural numbers in range
        if x is None:
            x=list(range(1,len(y)+1))
        
        #make the ticklabels string if they are numeric
        ticklabels=['$'+str(s) for s in ticklabels]
        
        plt.figure(0)
        #if big, resize figure
        if len(y)>10:
            plt.figure(figsize=(15, 10))
        #use colormap  
        my_cmap = plt.get_cmap("viridis")
        #plot bar
        plt.bar(x, y, tick_label = ticklabels,
                width = 0.8,color=my_cmap(self.rescale(y)))
        
        #add weight information
        for i in range(len(x)):
            plt.text(i+1,y[i]/2,str(y[i])+' kg',ha = 'center',
                 bbox = dict(facecolor = 'white', alpha =.9))
        # naming the x-axis
        plt.xlabel(xLabel)
        # naming the y-axis
        plt.ylabel(yLabel)
        # plot title
        plt.title(title)
        
        #axis management
        ax = plt.axes()
        ax.axes.get_yaxis().set_visible(False)
        #hide tick lines
        ax.tick_params(axis=u'both', which=u'both',length=0)
        ax.set_frame_on(False)

        # function to show the plot
        plt.show()
        
    #create bar chart with stacked bars on top of each other and capacity line
    def stackBarChart(self,y,values,title):
        #do not display unreasonably big data
        if(len(y)>30): return
        #switch to new figure
        #plt.figure(1)
        #make 2d if not
        if not isinstance(y[0], list):
            y=[y]
            values=[values]
            #add some padding in the graph
            plt.xlim([-0.8, 0.8])
            capX=0.8
        else:
             #resize graph
             plt.figure(figsize=(15, 10))
             capX=0.5*(len(y)-1)
        j=0
        #loop over y elements (2D list)
        while j < len(y):
            i=0
            #loop over y item (another list)
            for k in y[j]:
                #create bar graph in the same x location
                plt.bar(j, k, width = 0.8,color=np.random.rand(3,),bottom = sum(y[j][0:i]))
                i+=1
            i=0
            #loop over created bars and write their y value
            for k in range(len(y[j])):
                    plt.text(j,sum(y[j][0:i])+y[j][i]/2,str(y[j][i])+' kg',ha = 'center',
                         bbox = dict(facecolor = 'white', alpha =.9))
                    i+=1
            #add sum of the weights to the top
            plt.text(j,sum(y[j]),str(sum(y[j]))+' kg',ha = 'center',va='bottom',
                         bbox = dict(facecolor = 'yellow', alpha =.9))
            j+=1
        #axis management
        ax = plt.axes()
        plt.xticks(range(0,len(y)), ['$' +str(sum(x)) for x in values])
        plt.ylim([0,self.W+self.W/10])
        #hide tick lines
        ax.tick_params(axis=u'both', which=u'both',length=0)
        #hide y axis
        ax.axes.get_yaxis().set_visible(False)
        
        #capacity line line
        plt.axhline(y=self.W, color='r', linestyle='--')
        #capacity title through line
        plt.text(capX, self.W, 'capacity '+str(self.W) + 'kg', va='center', ha='center',backgroundcolor='white')
        # plot title
        plt.title(title)
         #ax1.axes.get_yaxis().set_visible(False)
        ax.set_frame_on(False)
        plt.show()
      

