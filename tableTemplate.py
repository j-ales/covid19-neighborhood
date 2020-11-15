# modified from http://cssmenumaker.com/br/blog/stylish-css-tables-tutorial
css = """
<style type=\"text/css\">
table {
color: #333;
font-family: Helvetica, Arial, sans-serif;
width: 1200px;
border-collapse:
collapse; 
border-spacing: 0;
}
td, th {
border: 1px solid transparent; /* No more visible border */
height: 30px;
}
th {
background: #DFDFDF; /* Darken header a bit */
font-weight: bold;
text-align: center;
}
td {
background: #EAEAEA;
text-align: center;
}
table tr:nth-child(odd) td{
background-color: white;
}
</style>
"""

def rankColorCSS(lastLevel4,lastLevel3,lastLevel2,lastLevel1):
    outputString = """
<style type=\"text/css\">
table {
color: #333;
font-family: Helvetica, Arial, sans-serif;
width: 1200px;
border-collapse:
collapse; 
border-spacing: 0;
}
td, th {
border: 1px solid transparent; /* No more visible border */
height: 30px;
}
th {
background: #DFDFDF; /* Darken header a bit */
font-weight: bold;
text-align: center;
}

td {
background: #EAEAEA;
text-align: center;
}

table tr:nth-child(odd) td{
background-color: white;
}
""" + f"""


table tr:nth-child(n+1):nth-child(-n+{lastLevel4}) td{{
color: #800080;
text-align: center;
}}

table tr:nth-child(n+{lastLevel4+1}):nth-child(-n+{lastLevel3}) td{{
color: #B22222;
text-align: center;
}}

table tr:nth-child(n+{lastLevel3+1}):nth-child(-n+{lastLevel2}) td{{
color: #FF4500;
text-align: center;
}}
table tr:nth-child(n+{lastLevel2+1}):nth-child(-n+{lastLevel1}) td{{
color: #228B22;
text-align: center;
}}

}}

</style>
"""
    return outputString


increaseCSS = """
<style type=\"text/css\">
table {
color: #7533;
font-family: Helvetica, Arial, sans-serif;
width: 1200px;
border-collapse:
collapse; 
border-spacing: 0;
}
td, th {
border: 1px solid transparent; /* No more visible border */
height: 30px;
}
th {
background: #d64b51; /* Darken header a bit */
font-weight: bold;
text-align: center;
}
td {
background: #ffcccc;
text-align: center;
}
table tr:nth-child(odd) td{
background-color: white;
}
</style>
"""

decreaseCSS = """
<style type=\"text/css\">
table {
color: #7533;
font-family: Helvetica, Arial, sans-serif;
width: 1200px;
border-collapse:
collapse; 
border-spacing: 0;
}
td, th {
border: 1px solid transparent; /* No more visible border */
height: 30px;
}
th {
background: #4bd651; /* Darken header a bit */
font-weight: bold;
text-align: center;
}
td {
background: #b6e3b9;
text-align: center;
}
table tr:nth-child(odd) td{
background-color: white;
}
</style>
"""