#!perl

# Process CIF files to get symmetry operators by simply importing and exporting through Materials Studio

use strict;
use Getopt::Long qw(GetOptions);
use File::Find qw(find);
use Cwd qw(getcwd);
use POSIX qw(ceil);
use MaterialsScript qw(:all);

use constant {
    PI => 4 * atan2(1, 1),
    BOXL => 28.0,
};

sub sine {
    return sin($_[0]*PI/180);
}

my %Args;
GetOptions(\%Args, "InputFolder=s", "OutputFolder=s");
my $inputfolder = $Args{InputFolder};
my $outputfolder = $Args{OutputFolder};

if ($inputfolder eq $outputfolder) {
    print "Output folder must be different to input folder!";
    exit(-1);
}

my ($cwd,$doc,$vol,$ax,$bx,$cx,$ah,$bh,$ch);
find {wanted => sub{
    if ($_ =~ /.*\.cif/) {
        $cwd=getcwd;
        print "$cwd/$_\n";
        $doc = Documents->Import("$cwd/$_");
        $doc->Export("$outputfolder/$_");
        #$doc->UnitCell->Atoms(0)->Style = "Ball and stick";
        $doc->CalculateBonds;
        $doc->SymmetryDefinition->OrientationConvention="A along X, B in XY plane";
        $vol = $doc->SymmetryDefinition->CellVolume;
        $ax = $doc->SymmetryDefinition->LengthA;
        $bx = $doc->SymmetryDefinition->LengthB;
        $cx = $doc->SymmetryDefinition->LengthC;
        $ah = $vol/($bx*$cx*sine($doc->SymmetryDefinition->AngleAlpha));
        $bh = $vol/($ax*$cx*sine($doc->SymmetryDefinition->AngleBeta));
        $ch = $vol/($bx*$ax*sine($doc->SymmetryDefinition->AngleGamma));
        $doc->BuildSuperCell(ceil(BOXL/$ah), ceil(BOXL/$bh), ceil(BOXL/$ch));
        # $doc->Export("$outputfolder/$_-small.pdb");
        # $doc->BuildSuperCell(2, 2, 2);
        # $doc->Export("$outputfolder/$_-large.pdb");
        $doc->Close;
    }
}},$inputfolder;

exit(0);