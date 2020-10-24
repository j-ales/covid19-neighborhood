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