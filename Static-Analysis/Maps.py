import numpy as np
import seaborn as sns
import folium
from folium.vector_layers import CircleMarker
from sklearn.preprocessing import MinMaxScaler
from branca.colormap import StepColormap
import json
import os

class Maps():

    def __init__(self,  Df_VF, dff_NT_map, dff_Ger_map, cenario, options, region=None):
        
        self.Df_VF  = Df_VF
        self.cenario =  cenario
        self.limit = {138: [0.95,1.05],230: [0.95,1.05],345: [0.95,1.05],440: [0.95,1.046],500: [1,1.10],525: [0.95,1.05],765: [0.90,1.046]}
        self.region = region
        
        file = os.path.abspath('Static-Analysis/RECURSOS/Coordenadas.json')
        with open(file, 'r') as json_file:
            self.state_geo = json.load(json_file)
        
        self.dff_Ger_map, self.dff_NT_map = self.get_df_LimitViolation(dff_NT_map, dff_Ger_map)
        # ======================================================================
        if options['Limit Violations All'] == True:
            self.get_Map_VL_Geral(['UHE','PCH','UTE','EOL','UFV',], Generation = True)
            self.get_Map_VL_Geral([ 765, 525, 500, 440, 345, 230], Generation = False)   
        # ======================================================================
        if options['Limit Violations by Group'] == True:
            self.get_Map_VLbygroup(['UHE','PCH','UTE','EOL','UFV'], Generation = True)
            self.get_Map_VLbygroup([765, 525, 500, 440, 345, 230], Generation = False)
        # ======================================================================
        if options['Mean and Variance'] == True:
            self.get_Map_Meanbygroup(['UHE','PCH','UTE','EOL','UFV'], Generation = True)
            self.get_Map_Meanbygroup([765, 525, 500, 440, 345, 230], Generation = False)
        # ======================================================================
        # ======================================================================
        if options['Limit Violations PO'] == True:
            self.get_Map_VL_PO(['UHE','PCH','UTE','EOL','UFV',], Generation = True)
            self.get_Map_VL_PO([ 765, 525, 500, 440, 345, 230], Generation = False)   

    def mapfeatures(self):
            map = folium.Map(
                location=[-14.2350, -51.9253],
                tiles='cartodbpositron',
                zoom_start=5,
                control_scale=True,
                min_zoom=4,
                max_zoom=19,
                max_bounds=True,
                max_bounds_viscosity=1.0,
                detect_retina=False,
                attribution='My Map',
                zoom_control=True,
                scroll_wheel_zoom=True,
                double_click_zoom=True,
                dragging=True,
                world_copy_jump=False,
                no_wrap=False,
                style='height: 100%; width: 100%; position: relative; left: 0; top: 0; z-index: 0; -webkit-user-drag: none;',
            )
            # Color the map
            folium.Choropleth(
                geo_data=self.state_geo,
                key_on='feature.properties.GEOCODIGO',
                fill_color='BuPu',  # Darker color scheme
                fill_opacity=0.4,   # Decreased transparency
                line_opacity=0.7,
                line_color='black',
                # legend_name='Voltage (pu)'
            ).add_to(map)

            return map

    def get_df_LimitViolation(self, dff_NT_map, dff_Ger_map, UniquePO = False):

        def contar_valores(valor_lista, valor_bajo, valor_alto, ):
            if UniquePO:
                valores_por_debajo = (1 if valor_lista < valor_bajo else 0 )
                valores_por_encima = (1 if valor_lista > valor_alto else 0)
            else:
                valores_por_debajo = sum(valor < valor_bajo for valor in valor_lista)
                valores_por_encima = sum(valor > valor_alto for valor in valor_lista)            
            return valores_por_debajo, valores_por_encima

        def df_limites(data, nt , limites, geracao =False):
            if geracao == False:
                indexbool = data['VBASEKV'] == nt
            else:
                indexbool = data['Gen_Type'] == nt
            # Aplicar la función a la columna 'MODV_PU' y almacenar los resultados en dos nuevas columnas 'valores_debajo' y 'valores_encima'
            data.loc[indexbool,'inferior'], data.loc[indexbool,'superior'] = zip(*data.loc[indexbool,'MODV_PU'].apply(lambda x: contar_valores(x, limites[0], limites[1])))
            return data

        # def df_countlimites(dff_NT_map, dff_Ger_map):
        dff_NT_map.loc[:, 'inferior'] = np.nan
        dff_NT_map.loc[:, 'superior'] = np.nan

        vbaseKV = list(set(dff_NT_map['VBASEKV'].unique()) & set(self.limit.keys()))
        for vbase in vbaseKV:
            df_limites(dff_NT_map, vbase ,self.limit[vbase])
        dff_Ger_map.loc[:, 'inferior'] = np.nan
        dff_Ger_map.loc[:, 'superior'] = np.nan
        gentype = list(set(dff_Ger_map['Gen_Type'].unique()) & set(['UHE', 'UTE', 'PCH', 'EOL', 'UFV', 'BIO']))
        
        for uf in gentype:
            df_limites(dff_Ger_map, uf ,[0.95, 1.05], True)

        return dff_Ger_map, dff_NT_map

    def get_Map_VLbygroup(self,lista, Generation):

        def asignar_color(Value):
            color1 = 'red'
            color2 = 'blue'
            if Value == 'inferior':
                return color2
            elif Value == 'superior':
                return color1
        
        color1 = 'red'
        color2 = 'blue'
        for uf in lista:

            map = self.mapfeatures()

            if Generation == True:
                Volt_geo = self.dff_Ger_map[self.dff_Ger_map['Gen_Type'] == uf]
            else:
                Volt_geo = self.dff_NT_map[self.dff_NT_map['VBASEKV'] == uf]


            for i in Volt_geo[['inferior',	'superior']].keys():

                df = Volt_geo[Volt_geo[i]!= 0]

                # Get latitude, longitude, voltage and other values
                Latitude = df['Latitude'].values
                Longitude = df['Longitude'].values
                Busnome = df['BUS_NAME'].values
                Busid = df['BUS_ID'].values
                numbarras = df[i].values
                if Generation == True:
                    Online = df['Online'].values
                    Compsyn = df['Compsync'].values
                    Sbase = df['BASE_MVA'].values
                    units = df['Ger_Units'].values

                try:
                    scaler = MinMaxScaler(feature_range=(2, 20))
                    barras_rshp = np.array(numbarras).reshape(-1, 1) # Reshape the variances array to fit the scaler input shape
                    longi = scaler.fit_transform(barras_rshp) # Normalize the variances using the MinMaxScaler
                    cant_barras = longi.flatten() # Flatten the normalized variances array

                    for idx, _ in  enumerate (Latitude):
                        if Generation == True:
                            caption = f"Name / ID : {Busnome[idx]} / {int(Busid[idx])} <br> N. vezes {i}: {int(numbarras[idx])} <br> Online : {int(Online[idx])} <br> Compsync : {int(Compsyn[idx])} <br> Sbase (MVA) : {int(Sbase[idx])} <br> U. Geradoras : {int(units[idx])}"
                        else:
                            caption = f"Name / ID : {Busnome[idx]} / {int(Busid[idx])} <br> Número_vezes_{i}: {int(numbarras[idx])} "
                        
                        CircleMarker(
                            location=[Latitude[idx], Longitude[idx]],
                            tooltip = caption,
                            radius=cant_barras[idx],
                            color='black',
                            fill=True,
                            fill_color=asignar_color(i),
                            fill_opacity=0.7,
                            weight=0.9
                        ).add_to(map)
                
                    # Add caption to the map (move to the upper-right border and change the format to bold)
                    if Generation == True:
                        caption_html = f'<div style="font-size: 14px; position: absolute; bottom: 25px; right: 20px; font-weight: bold;z-index: 9999; background-color: white; padding: 5px; border: 1px solid grey;"> Violações de Tensão - Cenário {self.cenario} <br> Barramento de Geração {uf} </div>'
                        legend_html = f'''
                            <div style="font-size: 14px; font-weight: bold; position: absolute; bottom: 80px; right: 20px; z-index: 9999; background-color: white; padding: 10px; border: 2px solid grey;">
                                <ul>
                                    <li><span style="color: {color1}">Num. Barras com Tensão Superior = {Volt_geo[Volt_geo['superior']!= 0]['superior'].count()} </span></p>
                                    <li><span style="color: {color2}">Num. Barras com Tensão Inferior = {Volt_geo[Volt_geo['inferior']!= 0]['inferior'].count()}</span></p>
                                <ul>
                            </div>
                        '''
                    else:
                        caption_html = f'<div style="font-size: 14px; position: absolute; bottom: 20px; right: 20px; font-weight: bold;z-index: 9999; background-color: white; padding: 5px; border: 1px solid grey;"> Violações de Tensão - Cenário {self.cenario} <br> Barramento Nivel de Tensão {uf} kV <br> Faixa Operacional ({self.limit[uf][0]} - {self.limit[uf][1]})</div>'
                        legend_html = f'''
                            <div style="font-size: 14px; font-weight: bold; position: absolute; bottom: 100px; right: 20px; z-index: 9999; background-color: white; padding: 10px; border: 2px solid grey;">
                                <ul>
                                    <li><span style="color: {color1}">Num. Barras com Tensão Superior = {Volt_geo[Volt_geo['superior']!= 0]['superior'].count()} </span></p>
                                    <li><span style="color: {color2}">Num. Barras com Tensão Inferior = {Volt_geo[Volt_geo['inferior']!= 0]['inferior'].count()}</span></p>
                                <ul>
                            </div>
                        '''
                        
                    map.get_root().html.add_child(folium.Element(caption_html))
                    map.get_root().html.add_child(folium.Element(legend_html))
                except:
                    print('no se tiene mapa de violação de limites para el nivel de tension ' + str(uf) + i) 

                archivo = self.cenario + '/Plots/Mapas/' + str(uf) + '_ViolaçãoLimites' + '.html'
                map.save(archivo)
        
    def get_Map_Meanbygroup(self, lista, Generation):

        if Generation == True:
            df_filtrado = self.dff_Ger_map[self.dff_Ger_map['Gen_Type'].isin(lista)]
        else:
            df_filtrado = self.dff_NT_map[self.dff_NT_map['VBASEKV'].isin(lista)]

        minimo = np.min([np.mean(volt) for volt in df_filtrado['MODV_PU'].values])
        maximo = np.max([np.mean(volt) for volt in df_filtrado['MODV_PU'].values])

        for uf in lista:
    
            map = self.mapfeatures()
            if Generation == True:
                Volt_geo = self.dff_Ger_map[self.dff_Ger_map['Gen_Type'] == uf]
            else:
                Volt_geo = self.dff_NT_map[self.dff_NT_map['VBASEKV'] == uf]
            # Get latitude, longitude, and voltage values
            Latitude = Volt_geo['Latitude'].values
            Longitude = Volt_geo['Longitude'].values
            Voltage = Volt_geo['MODV_PU'].values
            Busnome = Volt_geo['BUS_NAME'].values
            Busid = Volt_geo['BUS_ID'].values

            if Generation == True:
                Online = Volt_geo['Online'].values
                Compsyn = Volt_geo['Compsync'].values
                Sbase = Volt_geo['BASE_MVA'].values
                Pmax = Volt_geo['PMAX_MW'].values
                Pmin = Volt_geo['PMIN_MW'].values
                Qmax = Volt_geo['QMX_MVAR'].values
                Qmin = Volt_geo['QMN_MVAR'].values
                Pg = Volt_geo['PG_MW'].values
                Qg = Volt_geo['QG_MVAR'].values
                units = Volt_geo['Ger_Units'].values
            
            variances  = [np.std(lst) for lst in Volt_geo['MODV_PU']]
            # Create a MinMaxScaler instance
            scaler = MinMaxScaler(feature_range=(2, 20))
            var_reshaped = np.array(variances).reshape(-1, 1) # Reshape the variances array to fit the scaler input shape
            norm_variances = scaler.fit_transform(var_reshaped) # Normalize the variances using the MinMaxScaler
            norm_variances = norm_variances.flatten() # Flatten the normalized variances array
            colormap = StepColormap(colors=sns.color_palette("seismic"), vmin=minimo, vmax=maximo, max_labels=10)

            for idx, _ in  enumerate (Latitude):
                meanVolt = np.mean(Voltage[idx]).round(5)
                if Generation == True:  
                    caption = f"Name / ID : {Busnome[idx]} / {int(Busid[idx])} <br> V_mean / std (pu): {meanVolt} / ({format(variances[idx], '.3e')}) <br> Pg_mean / Qg_mean: {format(Pg[idx], '.1f')} / {format(Qg[idx],'.1f')} <br> Num. UG: {int(units[idx])} <br> S_base (MVA): {format(Sbase[idx], '.1f')}  <br> Pmax / Pmin: {format(Pmax[idx], '.1f')} / {format(Pmin[idx], '.1f')} <br> Qmax / Qmin: {format(Qmax[idx], '.1f')} / {format(Qmin[idx], '.1f')} <br> Online: {Online[idx]} <br> Compsync : {int(Compsyn[idx])}"
                else:
                    caption = f"Name / ID : {Busnome[idx]} / {int(Busid[idx])} <br> V_mean / std (pu): {meanVolt} / ({format(variances[idx], '.3e')})"
                CircleMarker(
                    location=[Latitude[idx], Longitude[idx]],
                    tooltip = caption,
                    # radius=norm_variances[idx],
                    radius=5,
                    color='black',
                    fill=True,
                    fill_color=colormap(meanVolt),
                    fill_opacity=0.8,
                    weight=1
                ).add_to(map)
        
            colormap.add_to(map)
            colormap.caption = 'Cenário ' + self.cenario + ( ' - Barramento Geração '+ str(uf) if Generation else ' - Barramento Nivel de Tensão '+ str(uf) + ' kV')
            archivo = self.cenario + '/Plots/Mapas/Mean_' + str(uf) + '.html'
            map.save(archivo)

    def get_Map_VL_Geral (self, lista, Generation):

        color1 = 'red'
        color2 = 'blue'
        map = self.mapfeatures()
        contsup = 0
        continf = 0 
        for i in ['inferior',	'superior']:

            for uf in lista:
                if Generation == True:
                    circle = False
                    Volt_geo = self.dff_Ger_map[self.dff_Ger_map['Gen_Type'] == uf]

                    if uf == 'EOL':
                        num_lados = 3
                    elif uf == 'UFV':
                        num_lados = 3
                    else:
                        circle = True
                else:
                    circle = False
                    Volt_geo = self.dff_NT_map[self.dff_NT_map['VBASEKV'] == uf]
                    if uf == 138:
                        num_lados = 3
                    elif uf == 230:
                        num_lados = 3
                    elif uf == 345:
                        num_lados = 3
                    elif uf == 440:
                        num_lados = 3
                    else:
                        circle = True

                df = Volt_geo[Volt_geo[i]!= 0]

                # Get latitude, longitude, voltage and other values
                Latitude = df['Latitude'].values
                Longitude = df['Longitude'].values
                Busnome = df['BUS_NAME'].values
                Busid = df['BUS_ID'].values
                numbarras = df[i].values
                if Generation == True:
                    Online = df['Online'].values
                    Compsyn = df['Compsync'].values
                    Sbase = df['BASE_MVA'].values
                    units = df['Ger_Units'].values

                if i == 'superior':
                    contsup = df['superior'].count() + contsup
                    colormap = StepColormap(colors=sns.color_palette("seismic",11)[-5:], vmin=0, vmax=1344, max_labels=10)#, max_labels=10
                elif i =='inferior':
                    continf = df['inferior'].count() + continf
                    colormap = StepColormap(colors=sns.color_palette("seismic",11)[:5][::-1], vmin=0, vmax=1344, max_labels=10)#, max_labels=10

                for idx, _ in  enumerate (Latitude):
                    if Generation == True:
                        caption = f"Name / ID : {Busnome[idx]} / {int(Busid[idx])} <br> N. vezes {i}: {int(numbarras[idx])} <br> Online : {int(Online[idx])} <br> Compsync : {int(Compsyn[idx])} <br> Sbase (MVA) : {int(Sbase[idx])} <br> U. Geradoras : {int(units[idx])}"
                    else:
                        caption = f"Name / ID : {Busnome[idx]} / {int(Busid[idx])} <br> Número_vezes_{i}: {int(numbarras[idx])} "
                    
                    if circle == True:
                        CircleMarker(
                            location=[Latitude[idx], Longitude[idx]],
                            tooltip = caption,
                            radius=8,
                            color=colormap(numbarras[idx]),
                            fill=True,
                            fill_color=colormap(numbarras[idx]),
                            fill_opacity=0.7,
                            weight=0.9
                        ).add_to(map)
                    else:
                        folium.features.RegularPolygonMarker(
                            location=[Latitude[idx], Longitude[idx]],
                            tooltip=caption,
                            fill_color=colormap(numbarras[idx]),
                            number_of_sides=num_lados,  # Number of sides for a square
                            radius=8,           # Size of the marker
                            rotation=30,         # Rotation angle for the square (0 means not rotated)
                            color=colormap(numbarras[idx]),
                            fill=True,
                            fill_opacity=0.7,
                            weight=0.9
                        ).add_to(map)

            colormap.add_to(map) 
            colormap.caption = ('Número vezes com tensão acima do limite operacional superior' if i == 'superior' else 'Número vezes com tensão abaixo do limite operacional inferior')
        # Add caption to the map (move to the upper-right border and change the format to bold)
        if Generation == True:
            savenome = 'Geracao'
            caption_html = f'<div style="font-size: 15px; position: absolute; bottom: 25px; right: 20px; font-weight: bold;z-index: 9999; background-color: white; padding: 5px; border: 1px solid grey;"> Violações de Tensão - Cenário {self.cenario} <br> Barramento de Geração (UHE, PCH , UTE, EOL, UFV) </div>'
            legend_html = f'''
                <div style="font-size: 15px; font-weight: bold; position: absolute; bottom: 80px; right: 20px; z-index: 9999; background-color: white; padding: 10px; border: 2px solid grey;">
                    <ul>
                        <li><span style="color: {color1}">Num. Barras com Tensão Superior = {contsup} </span></p>
                        <li><span style="color: {color2}">Num. Barras com Tensão Inferior = {continf} </span></p>
                    <ul>
                </div>
            '''
        else:
            savenome = 'NTensao'
            caption_html = f'<div style="font-size: 15px; position: absolute; bottom: 25px; right: 20px; font-weight: bold;z-index: 9999; background-color: white; padding: 5px; border: 1px solid grey;"> Violações de Tensão - Cenário {self.cenario} <br> Barramento Niveis de Tensão (765, 525, 500, 440, 345, 230) kV </div>'
            legend_html = f'''
                <div style="font-size: 15px; font-weight: bold; position: absolute; bottom: 80px; right: 20px; z-index: 9999; background-color: white; padding: 10px; border: 2px solid grey;">
                    <ul>
                        <li><span style="color: {color1}">Num. Barras com Tensão Superior = {contsup} </span></p>
                        <li><span style="color: {color2}">Num. Barras com Tensão Inferior = {continf} </span></p>
                    <ul>
                </div>
            '''
            
        map.get_root().html.add_child(folium.Element(caption_html))
        map.get_root().html.add_child(folium.Element(legend_html))

        archivo = self.cenario + '/Plots/Mapas/' + savenome + '_ViolaçãoLimites' + '.html'
        map.save(archivo)
        
    def get_Map_VL_PO(self, lista, Generation):
    
        def get_limitMapG_PO (Datadf,lista, Generation):
            color1 = 'red'
            color2 = 'blue'
            map = self.mapfeatures()
            contsup = 0
            continf = 0 
            for i in ['inferior',	'superior']:
                for uf in lista:
                    if Generation == True:
                        circle = False
                        Volt_geo = Datadf[Datadf['Gen_Type'] == uf]
                        
                        if uf == 'EOL':
                            num_lados = 3
                        elif uf == 'UFV':
                            num_lados = 3
                        else:
                            circle = True
                    else:
                        circle = False
                        Volt_geo = Datadf[Datadf['VBASEKV'] == uf]
                        if uf < 500:
                            num_lados = 3
                        else:
                            circle = True

                    df = Volt_geo[Volt_geo[i]!= 0]

                    df_lat_long = Volt_geo[(Volt_geo['inferior']!= 0) | (Volt_geo['superior']!= 0)][['Latitude','Longitude']].value_counts().to_frame()
                    Latitude = df['Latitude'].values
                    Longitude = df['Longitude'].values
                    Busnome = df['BUS_NAME'].values
                    voltage = df['MODV_PU'].values
                    Busid = df['BUS_ID'].values
                    numbarras = df[i].values

                    if Generation == True:
                        Online = df['Online'].values
                        Compsyn = df['Compsync'].values
                        Pmax = df['PMAX_MW'].values
                        Pmin = df['PMIN_MW'].values
                        Qmax = df['QMX_MVAR'].values
                        Qmin = df['QMN_MVAR'].values
                        Pg = df['PG_MW'].values
                        Qg = df['QG_MVAR'].values
                        units = df['Ger_Units'].values
                    else:
                        Shunt = df['SHUNT'].values
                        ShuntIndinst = df['SHUNT_INST_IND'].values
                        ShuntCapinst = df['SHUNT_INST_CAP'].values
                        
                    if Generation == True:
                        if i == 'superior':
                            contsup = df['superior'].count() + contsup
                            colormap = StepColormap(colors=sns.color_palette("seismic",11)[-5:], vmin=1.05, vmax=1.10, max_labels=10)#, max_labels=10
                        elif i =='inferior':
                            continf = df['inferior'].count() + continf
                            colormap = StepColormap(colors=sns.color_palette("seismic",11)[:5][::1], vmin=0.90, vmax=0.95, max_labels=10)#, max_labels=10
                    else:
                        if i == 'superior':
                            contsup = df['superior'].count() + contsup
                            colormap = StepColormap(colors=sns.color_palette("seismic",11)[-5:], vmin=self.limit[uf][1], vmax=1.10, max_labels=10)#, max_labels=10
                        elif i =='inferior':
                            continf = df['inferior'].count() + continf
                            colormap = StepColormap(colors=sns.color_palette("seismic",11)[:5][::1], vmin=0.90, vmax=self.limit[uf][0], max_labels=10)#, max_labels=10

                    for idx, _ in  enumerate (Latitude):

                        if type(voltage[idx]) == float:
                            volt = voltage[idx]
                        else:
                            volt = voltage[idx][0]
                        
                        num_lat_long = df_lat_long.loc[Latitude[idx],Longitude[idx]].values[0]
                        
                        if Generation == True:
                            # caption = f"Name / ID : {Busnome[idx]} / {int(Busid[idx])} <br> Voltage : {voltage[idx]}  <br> N. vezes {i}: {int(numbarras[idx])} <br> Online : {int(Online[idx])} <br> Compsync : {int(Compsyn[idx])} <br> Sbase (MVA) : {int(Sbase[idx])} <br> U. Geradoras : {int(units[idx])}"
                            caption = f"Name | ID : {Busnome[idx]} | {int(Busid[idx])} <br> V_pu: {volt} <br> Pg | Qg: {format(Pg[idx], '.1f')} | {format(Qg[idx],'.1f')} <br> Num. UG: {int(units[idx])} <br>  Pmax | Pmin: {format(Pmax[idx], '.1f')} | {format(Pmin[idx], '.1f')}  <br> Qmax | Qmin: {format(Qmax[idx], '.1f')} | {format(Qmin[idx], '.1f')} <br> U.G Online: {Online[idx]} <br> C.Sync Online : {int(Compsyn[idx])} <br> Lat|Long: {Latitude[idx]} | {Longitude[idx]} <br> Mesma Lat/Long: {num_lat_long} "
                        else:
                            caption = f"Name | ID : {Busnome[idx]} | {int(Busid[idx])} <br> Voltage : {volt}  <br>  Shunt: {Shunt[idx]}  <br>  ShuntIndinst: {ShuntIndinst[idx]}  <br>  ShuntCapinst: {ShuntCapinst[idx]} <br> Lat|Long: {Latitude[idx]} | {Longitude[idx]} <br> Mesma Lat/Long: {num_lat_long}"
                        
                        if circle == True:
                            CircleMarker(
                                location=[Latitude[idx], Longitude[idx]],
                                tooltip = caption,
                                radius=13,
                                color=colormap(volt),
                                fill=True,
                                fill_color=colormap(volt),
                                fill_opacity=0.5,
                                weight=2  # Adjust this to control the actual width of the border
                            ).add_to(map)
                        else:
                            folium.features.RegularPolygonMarker(
                                location=[Latitude[idx], Longitude[idx]],
                                tooltip=caption,
                                fill_color=colormap(volt),
                                number_of_sides=num_lados,  # Number of sides for a square
                                radius=13,           # Size of the marker
                                rotation=30,         # Rotation angle for the square (0 means not rotated)
                                color=colormap(volt),
                                fill=True,
                                fill_opacity=0.5,
                                weight=2  # Adjust this to control the actual width of the border
                            ).add_to(map)

                colormap.add_to(map) 
                colormap.caption = ('Tensão acima do limte operacional p.u' if i == 'superior' else 'Tensão abaixo do limte operacional p.u')
            # Add caption to the map (move to the upper-right border and change the format to bold)
            if Generation == True:
                savenome = 'DPI_PV_'
                caption_html = f'<div style="font-size: 15px; position: absolute; bottom: 25px; right: 20px; font-weight: bold;z-index: 9999; background-color: white; padding: 5px; border: 1px solid grey;"> Violações de Tensão - Cenário {self.cenario} <br> Barramento de Geração (UHE, PCH , UTE, EOL, UFV) </div>'
                legend_html = f'''
                    <div style="font-size: 15px; font-weight: bold; position: absolute; bottom: 80px; right: 20px; z-index: 9999; background-color: white; padding: 10px; border: 2px solid grey;">
                        <ul>
                            <li><span style="color: {color1}">Num. Barras com Tensão Superior = {contsup} </span></p>
                            <li><span style="color: {color2}">Num. Barras com Tensão Inferior = {continf} </span></p>
                        <ul>
                    </div>
                '''
            else:
                savenome = 'DPI_PQ_'
                caption_html = f'<div style="font-size: 15px; position: absolute; bottom: 25px; right: 20px; font-weight: bold;z-index: 9999; background-color: white; padding: 5px; border: 1px solid grey;"> Violações de Tensão - Cenário {self.cenario} <br> Barramento Niveis de Tensão (765, 525, 500, 440, 345, 230) kV </div>'
                legend_html = f'''
                    <div style="font-size: 15px; font-weight: bold; position: absolute; bottom: 80px; right: 20px; z-index: 9999; background-color: white; padding: 10px; border: 2px solid grey;">
                        <ul>
                            <li><span style="color: {color1}">Num. Barras com Tensão Superior = {contsup} </span></p>
                            <li><span style="color: {color2}">Num. Barras com Tensão Inferior = {continf} </span></p>
                        <ul>
                    </div>
                '''    
            map.get_root().html.add_child(folium.Element(caption_html))
            map.get_root().html.add_child(folium.Element(legend_html))
            archivo = self.cenario + '/Plots/Mapas/' + savenome + '.html'
            map.save(archivo)
            # display(map)

        regiao = self.region
        
        if Generation == True:
            data = self.dff_Ger_map[self.dff_Ger_map['Gen_Type'].isin(lista)]

            if regiao != None:
                data_ = data[data['REG'] == regiao]
            else:
                data_ = data

            get_limitMapG_PO(data_, ['UHE','PCH','UTE','EOL','UFV',], Generation = Generation)
        else:
            data = self.dff_NT_map[self.dff_NT_map['VBASEKV'].isin(lista)]

            if regiao != None:
                data_ = data[data['REG'] == regiao]
            else:
                data_ = data

            get_limitMapG_PO(data_, [765, 525, 500, 440, 345, 230], Generation = Generation)


# Df_VF = pd.read_csv('Df_VF.csv')
# dff_NT_map = pd.read_csv('dff_NT_map.csv')
# dff_Ger_map = pd.read_csv('dff_Ger_map.csv')
# cenario = 'V2A2F2_semUCT'
# options = {'Limit Violations All': True, 'Mean and Variance': False, 'Limit Violations by Group': False, 'HeatMap by state 1': False}
# Mapas(Df_VF, dff_NT_map, dff_Ger_map, cenario, options)