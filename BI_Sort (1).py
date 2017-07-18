# -*- coding: utf-8 -*-
import csv
fieldnames=['Dates','Category','Descript','DayOfWeek','PdDistrict','Resolution','Address','X','Y']
delCategory=['MISSING PERSON','RECOVERED VEHICLE','NON-CRIMINAL','RUNAWAY','SUICIDE','WARRANTS','SUSPICIOUS OCC','STOLEN PROPERTY']
delNachDiscrDrugs=['CONTROLLED SUBSTANCE VIOLATION, LOITERING FOR','LOITERING WHERE NARCOTICS ARE SOLD/USED','MAINTAINING PREMISE WHERE NARCOTICS ARE SOLD/USED','VISITING WHERE DRUGS ARE USED OR SMOKED ']
delNachDiscrOthers=['DOG, BARKING','DOG, FIGHTING','UNKNOWN COMPLAINT','INCIDENT ON SCHOOL GROUNDS','POSSESSION OF BURGLARY TOOLS','FALSE REPORT OF BOMB','MISCELLANEOUS INVESTIGATION','WEARING THE APPAREL OF OPPOSITE SEX TO DECEIVE','GUIDE DOG, INTERFERING WITH','FALSE REPORT OF CRIME','DOG, STRAY OR VICIOUS','DRIVERS LICENSE, SUSPENDED OR REVOKED','FALSE FIRE ALARM']
delSecondaryLaw_nachDeskr=['GANG ACTIVITY','JUVENILE INVOLVED','PREJUDICE-BASED INCIDENT']
wirtschaftsdelikte=['BAD CHECKS','BRIBERY','EMBEZZLEMENT','EXTORTION','FORGERY/COUNTERFEITING','FRAUD']
einbruch=['BURGLARY','ROBBERY','TREA','TRESPASS','VEHICLE THEFT']#mit gewahlt
diebstahl=['LARCENY/THEFT']#kleiner Diebstahl
beschaedigung_Gegenstaende=['ARSON','VANDALISM']
drogen_waffen=['DRIVING UNDER THE INFLUENCE','DRUG/NARCOTIC','LIQUOR LAWS','DRUNKENNESS','WEAPON LAWS']
sex_delikte=['PROSTITUTION','SEX OFFENSES FORCIBLE','SEX OFFENSES NON FORCIBLE','PORNOGRAPHY/OBSCENE MAT']
andere_delikte=['FAMILY OFFENSES','GAMBLING','OTHER OFFENSES','DISORDERLY CONDUCT','LOITERING']
From_secondaryLaw_disc_TOkoerperverletzung=['ASSAULT BY JUVENILE SUSPECT','BATTERY BY JUVENILE SUSPECT','DOMESTIC VIOLENCE','SHOOTING BY JUVENILE SUSPECT']
From_OtherDISCR_ToWirtschaftsdelikte=['DEFRAUDING TAXI DRIVER','OVERCHARGING TAXI FARE','FALSE PERSONATION TO RECEIVE MONEY OR PROPERTY','MONEY, PROPERTY OR LABOR, FRAUDULENTLY OBTAINING','INSURED PROPERTY/, DESTRUCTION TO DEFRAUD INSURER','JUDGE/JUROR ACCEPTING A BRIBE']
From_OtherDISCR_ToEinbruch_Diebstahl=['SCHOOL, PUBLIC, TRESPASS','LICENSE PLATE, STOLEN','LOST/STOLEN LICENSE PLATE ']
From_OtherDISCR_ToDrogen_Waffen=['INTOXICATED JUVENILE','OPEN CONTAINER OF ALCOHOL IN VEHICLE']
From_OtherDISCR_ToSex_delikte=['OBSCENE PHONE CALLS(S)','PHONE CALLS, OBSCENE']

with open('train.csv') as csvfile:
    newcsv= open('BI_new.csv', 'wb')
    reader=csv.DictReader(csvfile.read().decode('utf-8-sig').encode('utf-8').splitlines(), delimiter=',')
    writer = csv.DictWriter(newcsv, fieldnames=fieldnames, delimiter=',')
    writer.writeheader()
    for row in reader:
        if row['Category'] in wirtschaftsdelikte or(row['Descript']=="ATM RELATED CRIME" and row['Category']=="SECONDARY CODES")\
           or (row['Descript'] in From_OtherDISCR_ToWirtschaftsdelikte and row['Category']=="OTHER OFFENSES"):
           row['Category']="Wirtschaftsdelikte"
        if row['Category'] in einbruch or \
           (row['Descript'] in From_OtherDISCR_ToEinbruch_Diebstahl and row['Category']=="OTHER OFFENSES"):
            row['Category']="Einbruch/Raub"
        if row['Category'] in drogen_waffen or \
                (row['Descript'] in From_OtherDISCR_ToDrogen_Waffen and row['Category']=="OTHER OFFENSES") or\
                (row['Descript']=="WEAPONS POSSESSION BY JUVENILE SUSPECT" and row['Category']=="SECONDARY CODES"):
            row['Category'] ="Drogen-/Waffendelikte"
        if row['Category'] in sex_delikte or \
           (row['Descript'] in From_OtherDISCR_ToSex_delikte and row['Category']=="OTHER OFFENSES"):
            row['Category'] = "Sexualdelikte"
        if row['Category'] in andere_delikte:
            row['Category'] = "Andere Delikte"
        if row['Category']=="ASSAULT" or (row['Descript'] in From_secondaryLaw_disc_TOkoerperverletzung and row['Category']=="SECONDARY CODES"):
            row['Category']="Koerperverletzung"
        if row['Category'] in beschaedigung_Gegenstaende:
            row['Category']='Beschaedigung von Gegenstaenden'
        if row['Category'] in diebstahl:
            row['Category']='Diebstahl'
        if row['Category']in delCategory or (row['Descript'] in delNachDiscrDrugs and row['Category']=="DRUG/NARCOTIC") or \
                (row['Descript'] in delNachDiscrOthers and row['Category']=="OTHER OFFENSES") or (row['Descript'] in delSecondaryLaw_nachDeskr and row['Category']=="SECONDARY CODES"):
            del row
        else:
            writer.writerow(row)
