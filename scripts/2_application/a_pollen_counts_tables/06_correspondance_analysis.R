
# This sciprt generates the files produced from the correspondance analysis and used to plot the corresponding figures from the manuscript
# The generated files, used in the associated publication, are also available in the repository (4.Tables_for_figures/tables_AC_plots/)

library(lme4)

library(carData)
library(car)
library(FactoMineR)
library(conflicted)
library(dplyr)
library(factoextra)
library(ade4)
library(FactoMineR)
library(factoextra)


MONITORING_DATA =  "REPOSITORY/2.Application monitoring REPOSITORY"

fullpath <- file.path(
  MONITORING_DATA,
  "Tables_PollenCounts",
  "percentages_persiteperyear_forAC_withOther.csv" # percentages_persiteperyear_forAC_withOther.csv # percentages_persite_forAC_withOther.csv
)



li_sites <- c("W1", "W2", "W3", "W4", "D1", "D2", "D3")

mpath_fig = file.path(MONITORING_DATA, "Tables_PollenCounts/tables_AC_plots")
df <- read.csv(fullpath)

rownames(df) <- df$site # site_year or site !
df$site_year <- NULL  # site_year or site !
#df$site <- NULL 
ca_all <- CA(df, graph = FALSE)


# Coordinates
row_coords <- as.data.frame(ca_all$row$coord)
col_coords <- as.data.frame(ca_all$col$coord)
row_coords$site_year <- rownames(row_coords) # site_year or site !
#row_coords$site <- rownames(row_coords) # site_year or site !

site_pattern <- paste(li_sites, collapse="|")
row_coords$site <- regmatches(row_coords$site_year, regexpr(site_pattern, row_coords$site_year)) # site_year or site !
#row_coords$site <- regmatches(row_coords$site, regexpr(site_pattern, row_coords$site)) # site_year or site !

eig_df <- as.data.frame(ca_all$eig)


# Save files for Python (move the files in dedicated folders for persite and persiteperyear, to avoid overlaying new file)
write.csv(row_coords, file.path(mpath_fig, "row_coordinates.csv"), row.names = FALSE)
write.csv(col_coords, file.path(mpath_fig, "col_coordinates.csv"), row.names = TRUE)
write.csv(eig_df, file.path(mpath_fig, "eigenvalues.csv"))




# Circles --- 
coord_taxa <- ca_all$col$coord[, 1:2]  # Coordonnées des colonnes (taxa)
contrib_taxa <- ca_all$col$contrib[, 1:2]  # Contributions
radius_taxa <- sqrt(contrib_taxa[,1]^2 + contrib_taxa[,2]^2) * 0.1  # Ajuste le facteur

df_circles <- data.frame(
  taxa = rownames(coord_taxa),
  x = coord_taxa[,1],
  y = coord_taxa[,2], 
  radius = radius_taxa
)

write.csv(df_circles, file.path(mpath_fig, "circles.csv"), row.names = TRUE) # (move the files in dedicated folders for persite and persiteperyear, to avoid overlaying new file)


# Cos --- 
cos2_taxa <- ca_all$col$cos2[, 1:2]
df_cos2 <- data.frame(
  taxa = rownames(cos2_taxa),
  x = ca_all$col$coord[,1],
  y = ca_all$col$coord[,2],
  cos2_total = cos2_taxa[,1] + cos2_taxa[,2]
)
fullpath_cos <- 
write.csv(df_cos2, file.path(mpath_fig, "cos2_values.csv"), row.names = TRUE) # (move the files in dedicated folders for persite and persiteperyear, to avoid overlaying new file)






